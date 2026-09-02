use std::{
    borrow::Borrow,
    hash::{BuildHasher, Hash, RandomState},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicPtr, Ordering},
};

use seize::LocalGuard;

use crate::table::{DentTable, InsertResult, TableEntry, TrackingEntry};

pub(crate) mod probe;
pub(crate) mod table;

pub struct JanusMap<K, V, S = RandomState> {
    base: DentTable<K, V>,
    hasher: S,
}

impl<K, V, S> Default for JanusMap<K, V, S>
where
    K: Eq + Hash,
    S: Default + BuildHasher + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new_inner(16, S::default())
    }
}

impl<K: Eq + Hash, V: Clone> JanusMap<K, V, RandomState> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new_inner(capacity, RandomState::new())
    }
}

impl<K, V, S> JanusMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Returns the h1 and h2 hash for the given key.
    #[inline]
    fn hash<Q>(&self, key: &Q) -> (usize, u8)
    where
        Q: Hash + ?Sized,
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
}

impl<K: Eq + Hash, V: Clone, S: BuildHasher + Clone> JanusMap<K, V, S> {
    fn new_inner(capacity: usize, hasher: S) -> JanusMap<K, V, S> {
        let base = DentTable::<K, V>::new(capacity);
        JanusMap { base, hasher }
    }

    pub fn with_hasher(hasher: S) -> Self {
        JanusMap::with_capacity_and_hasher(32, hasher)
    }

    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
        Self::new_inner(capacity, hasher)
    }

    // TODO: Verify that the guard belongs to this map
    pub fn guard(&self) -> LocalGuard<'_> {
        self.base.collector.enter()
    }

    pub fn len(&self) -> usize {
        self.base.len.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.base.len.load(Ordering::Relaxed) == 0
    }

    pub fn insert(&self, key: K, value: V, guard: &LocalGuard<'_>) -> Option<V> {
        let (h1, h2) = self.hash(&key);
        match self.base.insert(key, h1, h2, value, true, guard) {
            InsertResult::Inserted => None,
            InsertResult::Replaced(val) => Some(val),
            InsertResult::Error => {
                unreachable!()
            }
        }
    }

    // pub fn try_insert(&self, key: K, value: V) -> Result<(), TryInsertError> {
    //     let (h1, h2) = self.hash(&key);
    //     match self.base.insert(key, h1, h2, value, false) {
    //         InsertResult::Inserted => Ok(()),
    //         InsertResult::Error => Err(TryInsertError::AlreadyExists),
    //         InsertResult::Replaced(_) => unreachable!(),
    //     }
    // }

    pub fn get<'g, Q>(&self, key: &Q, guard: &'g LocalGuard<'_>) -> Option<ReadGuard<'g, V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (h1, h2) = self.hash(key);
        self.base.get(h1, h2, guard)
    }

    pub fn get_mut<'g, Q>(
        &'g self,
        key: &Q,
        guard: &'g LocalGuard<'_>,
    ) -> Option<WriteGuard<'g, K, V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (h1, h2) = self.hash(key);
        self.base.get_mut(h1, h2, guard)
    }

    pub fn remove<Q>(&self, key: &Q, guard: &LocalGuard<'_>) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (h1, h2) = self.hash(key);
        self.base.remove(h1, h2, guard)
    }
}

impl<K, V, S> Drop for JanusMap<K, V, S> {
    fn drop(&mut self) {
        let table = self.base.inner.load(Ordering::Acquire);
        if table.is_null() {
            return;
        }
        DentTable::deallocate(table);
    }
}

#[derive(Debug)]
pub enum TryInsertError {
    AlreadyExists,
}

pub struct ReadGuard<'a, V> {
    entry: *mut TrackingEntry<V>,
    _keep_alive: PhantomData<&'a ()>,
}

impl<'a, V> ReadGuard<'a, V> {
    fn as_ref(&self) -> &V {
        &unsafe { &mut (*self.entry) }.target
    }
}

impl<'a, V> Deref for ReadGuard<'a, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a, V> Drop for ReadGuard<'a, V> {
    fn drop(&mut self) {
        unsafe { &(*self.entry) }.drop_reader();
    }
}

// TODO
// It should be possible to allow the entry to
// get removed while write is active
// But WriteGuard needs to hold both AtomicPtr<TableEntry> and *mut TableEntry
// Then during drop check if AtomicPtr holds null
pub struct WriteGuard<'a, K, V> {
    // We are the only owner of this pointer and data behind it
    write_ptr: *mut TrackingEntry<V>,
    slot: &'a AtomicPtr<TableEntry<K, V>>,
    _keep_alive: PhantomData<&'a ()>,
}

impl<'a, K, V> WriteGuard<'a, K, V> {
    fn as_ref(&self) -> &V {
        &unsafe { &mut (*self.write_ptr) }.target
    }

    fn as_mut_ref(&mut self) -> &mut V {
        &mut unsafe { &mut (*self.write_ptr) }.target
    }
}

impl<'a, K, V> Deref for WriteGuard<'a, K, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a, K, V> DerefMut for WriteGuard<'a, K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_ref()
    }
}

impl<'a, K, V> Drop for WriteGuard<'a, K, V> {
    fn drop(&mut self) {
        // Guaranteed to be non-null for as long as we have write access
        let slot_ptr = self.slot.load(Ordering::Acquire);
        let old_read_ptr =
            unsafe { &(*slot_ptr) }.access[0].swap(self.write_ptr, Ordering::Release);
        unsafe { &(*slot_ptr) }.access[1].swap(old_read_ptr, Ordering::Release);
    }
}

#[cfg(test)]
mod test {
    use crate::JanusMap;

    #[test]
    fn basic_test() {
        let map = JanusMap::<u64, String>::with_capacity(16);
        let guard = map.guard();
        map.insert(1, "aaa".into(), &guard);
        println!("map len: {}", map.len());
        if let Some(old) = map.insert(1, "bbb".into(), &guard) {
            eprintln!("Old value: {old}")
        }
        println!("map len: {}", map.len());

        let entry = map.get_mut(&1, &guard).unwrap();
        let value = entry.as_ref();
        eprintln!("get entry: {}", value);
        drop(entry);

        map.remove(&1, &guard);

        // eprintln!("length: {}", map.len());
        // map.insert(2, "aaa".into());
        // eprintln!("length: {}", map.len());
    }
}
