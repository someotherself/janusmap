use std::{
    alloc::{Layout, handle_alloc_error},
    borrow::Borrow,
    cell::UnsafeCell,
    hash::{BuildHasher, Hash, RandomState},
    marker::PhantomData,
    mem::MaybeUninit,
    sync::atomic::{AtomicPtr, AtomicU8, AtomicU16, AtomicUsize, Ordering},
};

use crossbeam::utils::CachePadded;
use seize::{Collector, Guard, LocalGuard};

use crate::{
    ReadGuard, WriteGuard,
    probe::{self, Probe},
};

pub(crate) mod hash_metadata {
    pub const EMPTY_SLOT: u8 = 0;
    /// Indicates that the entry was removed from this slot
    ///
    /// But the slot may be re-used in the future because writes are serialized
    pub const TOMBSTONE_SLOT: u8 = 0x80;
}

pub struct DentTable<K, V, S = RandomState> {
    pub(crate) len: CachePadded<AtomicUsize>,
    #[allow(unused)]
    pub(crate) capacity: usize,
    pub(crate) collector: Collector,
    pub(crate) hasher: S,
    pub(crate) inner: AtomicPtr<RawTable<K, V>>,
}

// #[repr(transparent)]
// pub struct MapGuard<G>(G);

// impl<G> MapGuard<G> {
//     /// Create a new `MapGuard`.
//     ///
//     /// # Safety
//     ///
//     /// The guard must be valid to use with the given map.
//     pub unsafe fn new(guard: G) -> MapGuard<G> {
//         MapGuard(guard)
//     }

//     /// Create a new `MapGuard` from a reference.
//     ///
//     /// # Safety
//     ///
//     /// The guard must be valid to use with the given map.
//     pub unsafe fn from_ref(guard: &G) -> &MapGuard<G> {
//         // Safety: `VerifiedGuard` is `repr(transparent)` over `G`.
//         unsafe { &*(guard as *const G as *const MapGuard<G>) }
//     }
// }

impl<K, V, S> DentTable<K, V, S> {
    /// Returns the h1 and h2 hash for the given key.
    #[inline]
    fn hash<Q>(&self, key: &Q) -> (usize, u8)
    where
        Q: Hash + ?Sized,
        S: BuildHasher + Clone,
    {
        let hash = self.hasher.hash_one(key);
        (Self::h1(hash), Self::h2(hash))
    }

    // Returns the primary hash for an entry.
    #[inline]
    fn h1(hash: u64) -> usize {
        hash as usize
    }

    /// Return a byte of hash metadata, used for cheap searches.
    #[inline]
    fn h2(hash: u64) -> u8 {
        const MIN_HASH_LEN: usize = if std::mem::size_of::<usize>() < std::mem::size_of::<u64>() {
            std::mem::size_of::<usize>()
        } else {
            std::mem::size_of::<u64>()
        };

        // Grab the top 7 bits of the hash.
        //
        // While the hash is normally a full 64-bit value, some hash functions
        // (such as fxhash) produce a usize result instead, which means that the
        // top 32 bits are 0 on 32-bit platforms.
        let top7 = hash >> (MIN_HASH_LEN * 8 - 7);

        // zero is reserved for empty slots
        (top7 & 0x7f) as u8 + 1
    }

    #[inline]
    pub fn new(capacity: usize, hasher: S) -> DentTable<K, V, S>
    where
        V: Clone,
    {
        let mut cap = capacity;
        if cap == 0 {
            cap = 16
        };
        let cap = cap.checked_mul(8).expect("capacity overflow") / 6;
        let cap = cap.next_power_of_two();

        let table = Self::allocate(cap);

        DentTable {
            len: CachePadded::new(AtomicUsize::new(0)),
            capacity,
            collector: Collector::new(),
            hasher,
            inner: AtomicPtr::new(table),
        }
    }

    pub(crate) fn deallocate(table: *mut RawTable<K, V>) {
        assert!(!table.is_null());

        let capacity = unsafe { &(*table) }.mask + 1;
        let (layout, short_hash_offset, entries_offset) = Self::layout(capacity);

        let base = table.cast::<u8>();

        let short_hash = unsafe { base.add(short_hash_offset) }.cast::<AtomicU8>();

        let entries = unsafe { base.add(entries_offset) }.cast::<AtomicPtr<TableEntry<K, V>>>();

        // First destroy/reclaim all live TableEntry allocations.
        //
        // This assumes each TableEntry was separately allocated and that
        // reclaim_entry deallocates the TableEntry allocation.
        for i in 0..capacity {
            let entry = unsafe { &(*entries.add(i)) }.load(Ordering::Relaxed);

            if !entry.is_null() {
                // reclaim_entry moves out the owned key/value and deallocates
                // the TableEntry allocation. Dropping the returned tuple drops
                // K and V.
                unsafe { drop(TableEntry::free_entry(entry)) };

                // The slot itself is no longer needed.
            }
        }

        // Destroy the trailing AtomicPtr objects.
        for i in 0..capacity {
            unsafe { std::ptr::drop_in_place(entries.add(i)) };
        }

        // Destroy the trailing AtomicU8 objects.
        for i in 0..capacity {
            unsafe { std::ptr::drop_in_place(short_hash.add(i)) };
        }

        // Destroy the RawTable header.
        //
        // Its zero-length-array fields do not touch the trailing arrays.
        unsafe { std::ptr::drop_in_place(table) };

        // `layout` must be exactly the layout used by `alloc`.
        unsafe { std::alloc::dealloc(base, layout) };
    }

    #[inline]
    fn allocate(capacity: usize) -> *mut RawTable<K, V>
    where
        V: Clone,
    {
        assert!(capacity.is_power_of_two());

        let buckets = capacity;
        let mask = buckets - 1;

        let (layout, short_hash_offset, entries_offset) = Self::layout(capacity);

        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        let table = ptr.cast::<RawTable<K, V>>();

        unsafe {
            std::ptr::write(
                table,
                RawTable {
                    mask,
                    probe_limit: probe::limit(mask),
                    short_hash: [],
                    entries: [],
                },
            );

            // init short_hash
            let short_hash = ptr.add(short_hash_offset).cast::<AtomicU8>();
            for i in 0..capacity {
                std::ptr::write(short_hash.add(i), AtomicU8::new(hash_metadata::EMPTY_SLOT));
            }

            // init entries
            let entries = ptr
                .add(entries_offset)
                .cast::<AtomicPtr<TableEntry<K, V>>>();
            for i in 0..capacity {
                std::ptr::write(entries.add(i), AtomicPtr::new(std::ptr::null_mut()));
            }
        }

        table
    }

    /// Returns a reference to the root hash-table.
    #[inline]
    fn root(&self, guard: &impl Guard) -> *mut RawTable<K, V> {
        // Load the root table.
        guard.protect(&self.inner, Ordering::Acquire)
    }

    #[inline]
    fn layout(cap: usize) -> (Layout, usize, usize) {
        let header = Layout::new::<RawTable<K, V>>();
        let shorts = Layout::array::<AtomicU8>(cap).unwrap();
        let entries = Layout::array::<AtomicPtr<TableEntry<K, V>>>(cap).unwrap();

        let (layout, short_hash_offset) = header.extend(shorts).unwrap();
        let (layout, entries_offset) = layout.extend(entries).unwrap();

        (layout.pad_to_align(), short_hash_offset, entries_offset)
    }

    #[inline]
    pub(crate) fn remove<Q>(&self, key: &Q, guard: &LocalGuard<'_>) -> Option<V>
    where
        V: Clone,
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
        S: BuildHasher + Clone,
    {
        let (h1, h2) = self.hash(key);
        let table = self.root(guard);
        let mask = Self::mask(table);
        let mut probe = Probe::start(h1, mask);

        loop {
            let short_hash = Self::short_hash(table, probe.i);

            if short_hash == h2 {
                match self.remove_entry(self.get_entry(probe.i), key, h1, guard) {
                    Some(val) => {
                        self.len.fetch_sub(1, Ordering::Relaxed);
                        Self::store_short_hash(table, probe.i, hash_metadata::TOMBSTONE_SLOT);
                        return Some(val);
                    }
                    None => {
                        probe.next(mask);
                        continue;
                    }
                }
            } else if short_hash == 0 {
                return None;
            } else {
                // slot is occupied by another entry
                // keep probing
                probe.next(mask);
                continue;
            }
        }
    }

    #[inline]
    fn remove_entry<Q>(
        &self,
        slot: &AtomicPtr<TableEntry<K, V>>,
        key: &Q,
        h1: usize,
        guard: &LocalGuard<'_>,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
        V: Clone,
    {
        let entry_ptr = slot.load(Ordering::Acquire);
        if entry_ptr.is_null() {
            // Another thread has removed the entry
            return None;
        }

        let entry_ref = unsafe { &(*entry_ptr) };
        if entry_ref.key.borrow() != key || entry_ref.hash != h1 {
            return None;
        }

        // Try to swap out the writer

        let mut write_ptr = entry_ref.access[1].swap(std::ptr::null_mut(), Ordering::Relaxed);
        loop {
            if write_ptr.is_null() {
                // writer already active
                // use parking_lot to wait?
                write_ptr = entry_ref.access[1].swap(std::ptr::null_mut(), Ordering::Relaxed);
                continue;
            }
            break;
        }

        // No writer active, but there may be readers stil active
        // and we also have a write lock on the entry.
        // Readers will drop when all the map guards are dropped

        // Swap out the entry entirely and then put back the write ptr

        let removed_slot = slot.swap(std::ptr::null_mut(), Ordering::Relaxed);

        if write_ptr.is_null() {
            // Already removed. Is it even possible at this point?
            // Should we swap slot and then writer? Returning None is probably bad
            return None;
        }

        let value = unsafe { &*write_ptr }.target.clone();

        unsafe { &(*removed_slot) }.access[1].store(write_ptr, Ordering::Relaxed);

        unsafe {
            guard.defer_retire(removed_slot, |ptr, _| {
                TableEntry::free_entry(ptr);
            })
        };

        Some(value)
    }

    #[inline]
    pub(crate) fn get_mut<'g, Q>(
        &'g self,
        key: &Q,
        guard: &'g LocalGuard<'_>,
    ) -> Option<WriteGuard<'g, K, V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
        S: BuildHasher + Clone,
    {
        let (h1, h2) = self.hash(key);
        let table = self.root(guard);

        let mut probe = Probe::start(h1, Self::mask(table));
        loop {
            let short_hash = Self::short_hash(table, probe.i);

            if short_hash == h2 {
                let slot = self.get_entry(probe.i);

                let entry_ptr = guard.protect(slot, Ordering::Acquire);

                if entry_ptr.is_null() {
                    // Entry was removed by another thread
                    probe.next(Self::mask(table));
                    continue;
                }

                let entry_ref = unsafe { &(*entry_ptr) };
                if entry_ref.key.borrow() != key || entry_ref.hash != h1 {
                    // Entry does not match
                    probe.next(Self::mask(table));
                    continue;
                }

                let entry_ref = unsafe { &(*entry_ptr) };

                // Try and aquire write access
                let mut write_ptr =
                    entry_ref.access[1].swap(std::ptr::null_mut(), Ordering::Relaxed); // Relaxed is probably not correct here

                loop {
                    if write_ptr.is_null() {
                        // writer already active
                        // TODO: use parking_lot to wait?
                        write_ptr =
                            entry_ref.access[1].swap(std::ptr::null_mut(), Ordering::Relaxed); // Relaxed is probably not correct here
                        continue;
                    }
                    break;
                }

                // We have a write lock
                // Check and wait for any active readers
                let readers = &unsafe { &(*write_ptr) }.readers;
                loop {
                    if readers.load(Ordering::Relaxed) != 0 {
                        // Relaxed is probably not correct here
                        // TODO: use parking_lot to wait?
                        continue;
                    }
                    break;
                }

                return Some(WriteGuard {
                    write_ptr,
                    slot,
                    _keep_alive: PhantomData,
                });
            } else if short_hash == 0 {
                // Key does not exist
                return None;
            } else {
                // slot is occupied by another entry
                // keep probing
                probe.next(Self::mask(table));
                continue;
            }
        }
    }

    #[inline]
    pub(crate) fn get<'g, Q>(&self, key: &Q, guard: &'g LocalGuard<'_>) -> Option<ReadGuard<'g, V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
        S: BuildHasher + Clone,
    {
        let (h1, h2) = self.hash(key);
        let table = self.root(guard);
        let mask = Self::mask(table);
        let mut probe = Probe::start(h1, mask);
        loop {
            let short_hash = Self::short_hash(table, probe.i);

            if short_hash == h2 {
                let slot = self.get_entry(probe.i);

                let entry_ptr = guard.protect(slot, Ordering::Acquire);

                if entry_ptr.is_null() {
                    // Entry was removed by another thread
                    probe.next(mask);
                    continue;
                }

                let entry_ref = unsafe { &(*entry_ptr) };
                if entry_ref.key.borrow() != key || entry_ref.hash != h1 {
                    // Entry does not match
                    probe.next(mask);
                    continue;
                }

                let entry_ref = unsafe { &(*entry_ptr) };

                // We now have the access to the read size
                let read_ptr = entry_ref.access[0].load(Ordering::Acquire);
                debug_assert!(!read_ptr.is_null());

                // Increment the read count
                unsafe { &*read_ptr }
                    .readers
                    .fetch_add(1, Ordering::Release);

                return Some(ReadGuard {
                    entry: read_ptr,
                    _keep_alive: PhantomData,
                });
            } else if short_hash == 0 {
                // Key does not exist
                return None;
            } else {
                // slot is occupied by another entry
                // keep probing
                probe.next(mask);
                continue;
            }
        }
    }

    #[inline]
    pub(crate) fn insert(
        &self,
        key: K,
        value: V,
        _replace: bool,
        guard: &impl Guard,
    ) -> InsertResult<V>
    where
        V: Clone,
        K: Eq + Hash,
        S: BuildHasher + Clone,
    {
        let (h1, h2) = self.hash(&key);
        let mut key = Some(key);
        let mut value = Some(value);

        let table = self.root(guard);
        let mask = Self::mask(table);
        let mut probe = Probe::start(h1, mask);

        loop {
            let short_hash = Self::short_hash(table, probe.i);
            if short_hash == 0 {
                // slot is empty
                // entry already exists
                let k = key.unwrap();
                let v = value.unwrap();
                match self.insert_new(probe.i, k, v, h1, h2, table) {
                    InsertNew::Found { k, v } => {
                        key = Some(k);
                        value = Some(v);
                        probe.next(mask);
                        continue;
                    }
                    InsertNew::Inserted => {
                        self.len.fetch_add(1, Ordering::Relaxed);
                        return InsertResult::Inserted;
                    }
                }
            } else if short_hash == h2 {
                let v = value.unwrap();
                let k = key.unwrap();
                match self.insert_replace(self.get_entry(probe.i), &k, v, true, h1) {
                    InsertReplace::Failed { v } => {
                        value = Some(v);
                        key = Some(k);
                        probe.next(mask);
                        continue;
                    }
                    InsertReplace::Replaced { value } => return InsertResult::Replaced(value),
                    InsertReplace::Found => return InsertResult::Error,
                }
            } else {
                // slot is occupied by another entry
                // keep probing
                probe.next(mask);
                continue;
            }
        }
    }

    #[inline]
    fn insert_new(
        &self,
        slot: usize,
        key: K,
        value: V,
        h1: usize,
        h2: u8,
        table: *mut RawTable<K, V>,
    ) -> InsertNew<K, V>
    where
        V: Clone,
    {
        let fresh_entry = self.get_entry(slot);
        let new_entry = TableEntry::new(key, h1, value);

        match fresh_entry.compare_exchange(
            std::ptr::null_mut(),
            new_entry,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                Self::store_short_hash(table, slot, h2);
                InsertNew::Inserted
            }
            Err(_) => {
                // Already claimed. Keep probing
                let (key, value) = unsafe { TableEntry::reclaim_entry(new_entry) };
                InsertNew::Found { k: key, v: value }
            }
        }
    }

    #[inline]
    fn insert_replace(
        &self,
        slot: &AtomicPtr<TableEntry<K, V>>,
        key: &K,
        value: V,
        replace: bool,
        h1: usize,
    ) -> InsertReplace<V>
    where
        K: Eq,
        V: Clone,
    {
        let entry_ptr = slot.load(Ordering::Acquire);
        if entry_ptr.is_null() {
            // Another thread has removed the entry
            return InsertReplace::Failed { v: value };
        }

        let entry_ref = unsafe { &(*entry_ptr) };
        if entry_ref.key != *key || entry_ref.hash != h1 {
            return InsertReplace::Failed { v: value };
        };
        if !replace {
            return InsertReplace::Found;
        }

        // We only need to replace the write target
        let entry_ref = unsafe { &(*entry_ptr) };
        loop {
            let write_ptr = entry_ref.access[1].swap(std::ptr::null_mut(), Ordering::Relaxed); // Relaxed is probably not correct here

            if write_ptr.is_null() {
                // writer already active
                // TODO: use parking_lot to wait?
                continue;
            }

            // We have a write lock
            // Check and wait for any active readers
            let readers = &unsafe { &(*write_ptr) }.readers;
            if readers.load(Ordering::Relaxed) != 0 {
                // Relaxed is probably not correct here
                // TODO: use parking_lot to wait?
                entry_ref.access[1].swap(write_ptr, Ordering::Relaxed);
                continue;
            }

            let old_value = &mut unsafe { &mut (*write_ptr) }.target;
            let value = value;
            let old = std::mem::replace(old_value, value);

            // Swap the write ptr with the read ptr. New value will not be available to readers
            let read_ptr = entry_ref.access[0].swap(write_ptr, Ordering::Relaxed);

            // Swap the read ptr in the write slot
            let null_ptr = entry_ref.access[1].swap(read_ptr, Ordering::Relaxed);
            debug_assert!(null_ptr.is_null());

            return InsertReplace::Replaced { value: old };
        }
    }

    #[inline]
    fn entries_ptr(&self, ptr: *mut RawTable<K, V>) -> *mut AtomicPtr<TableEntry<K, V>> {
        let capacity = unsafe { (*ptr).mask + 1 };
        let (_, _, entries_offset) = Self::layout(capacity);

        unsafe { ptr.cast::<u8>().add(entries_offset) }.cast::<AtomicPtr<TableEntry<K, V>>>()
    }

    #[inline]
    pub fn get_entry(&self, i: usize) -> &AtomicPtr<TableEntry<K, V>> {
        let ptr = self.inner.load(Ordering::Acquire);
        let entries = self.entries_ptr(ptr);

        unsafe { &*entries.add(i) }
    }

    #[inline]
    pub fn mask(ptr: *mut RawTable<K, V>) -> usize {
        unsafe { (&*ptr).mask }
    }

    #[inline]
    fn short_hash(ptr: *mut RawTable<K, V>, slot: usize) -> u8 {
        let ptr = unsafe { ptr.cast::<u8>().add(size_of::<RawTable<K, V>>()) }.cast::<AtomicU8>();
        unsafe { &(*ptr.add(slot)) }.load(Ordering::Acquire)
    }

    #[inline]
    fn store_short_hash(ptr: *mut RawTable<K, V>, slot: usize, h2: u8) {
        let ptr = unsafe { ptr.cast::<u8>().add(size_of::<RawTable<K, V>>()) }.cast::<AtomicU8>();
        unsafe { &(*ptr.add(slot)) }.store(h2, Ordering::Release);
    }
}

#[repr(C)]
pub struct RawTable<K, V> {
    pub(crate) mask: usize,
    pub(crate) probe_limit: usize,
    // Fixed header ends here
    pub(crate) short_hash: [AtomicU8; 0],
    pub(crate) entries: [AtomicPtr<TableEntry<K, V>>; 0],
}

pub(crate) struct TableEntry<K, V> {
    key: K,
    hash: usize,
    targets: [UnsafeCell<TrackingEntry<V>>; 2],
    // access[0] => always the read pointer
    // access[1] => always the write pointer
    // Access (the pointer) is swapped after every write
    //
    // When a writer tries to acquire, it will try to swap with null
    // If already null, another writer is already active
    pub(crate) access: [AtomicPtr<TrackingEntry<V>>; 2],
}

pub(crate) struct TrackingEntry<V> {
    pub(crate) target: V,
    // Placeholder until I figure out something better
    pub(crate) readers: AtomicU16,
}

impl<V> TrackingEntry<V> {
    pub(crate) fn drop_reader(&self) {
        self.readers.fetch_sub(1, Ordering::Release);
    }
}

impl<K, V> TableEntry<K, V> {
    fn new(key: K, hash: usize, value: V) -> *mut TableEntry<K, V>
    where
        V: Clone,
    {
        Self::alloc_entry(key, hash, value)
    }

    fn alloc_entry(key: K, hash: usize, value: V) -> *mut TableEntry<K, V>
    where
        V: Clone,
    {
        let layout = Layout::new::<MaybeUninit<TableEntry<K, V>>>();

        let raw = unsafe { std::alloc::alloc(layout).cast::<MaybeUninit<TableEntry<K, V>>>() };
        let raw =
            std::ptr::NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));

        let entry = raw.as_ptr().cast::<TableEntry<K, V>>();

        unsafe {
            std::ptr::addr_of_mut!((*entry).key).write(key);
            std::ptr::addr_of_mut!((*entry).hash).write(hash);
            std::ptr::addr_of_mut!((*entry).targets).write([
                UnsafeCell::new(TrackingEntry {
                    target: value.clone(),
                    readers: AtomicU16::new(0),
                }),
                UnsafeCell::new(TrackingEntry {
                    target: value,
                    readers: AtomicU16::new(0),
                }),
            ]);

            let left = (*entry).targets[0].get();
            let right = (*entry).targets[1].get();

            std::ptr::addr_of_mut!((*entry).access)
                .write([AtomicPtr::new(left), AtomicPtr::new(right)]);
        }

        entry
    }

    /// SAFETY:
    ///
    /// - The K and V returned need manual de-allocation
    /// - entry should be a non-null pointer
    /// - If key is already moved out, it will return None
    unsafe fn reclaim_entry(entry: *mut TableEntry<K, V>) -> (K, V)
    where
        V: Clone,
    {
        let layout = Layout::new::<MaybeUninit<TableEntry<K, V>>>();

        let key = unsafe { std::ptr::read(std::ptr::addr_of!((*entry).key)) };

        unsafe {
            let left = (*entry).targets[0].get();
            let right = (*entry).targets[1].get();

            let v = std::ptr::read(left).target;
            std::ptr::drop_in_place(right);

            std::alloc::dealloc(entry.cast::<u8>(), layout);

            (key, v)
        }
    }

    /// SAFETY:
    ///
    /// - entry should be a non-null pointer
    pub unsafe fn free_entry(entry: *mut TableEntry<K, V>) -> Option<K> {
        if entry.is_null() {
            return None;
        };
        let layout = Layout::new::<MaybeUninit<TableEntry<K, V>>>();

        // Should check if
        let key_ptr = std::ptr::addr_of!(unsafe { &(*entry) }.key);

        let key = if key_ptr.is_null() {
            None
        } else {
            Some(unsafe { std::ptr::read(std::ptr::addr_of!((*entry).key)) })
        };

        unsafe {
            let left_ptr = (*entry).targets[0].get();
            let right_ptr = (*entry).targets[1].get();

            std::ptr::drop_in_place(right_ptr);
            std::ptr::drop_in_place(left_ptr);

            std::alloc::dealloc(entry.cast::<u8>(), layout);
        }
        key
    }
}

pub enum InsertResult<V> {
    Inserted,
    Replaced(V),
    Error,
}

enum InsertNew<K, V> {
    Inserted,
    // Entry was inserted by a different thread
    Found { k: K, v: V },
}

enum InsertReplace<V> {
    Replaced { value: V },
    // Entry was removed by another thread
    Failed { v: V },
    Found,
}
