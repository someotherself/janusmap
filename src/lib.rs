use std::{
    borrow::Borrow,
    hash::{BuildHasher, Hash, RandomState},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::atomic::Ordering,
};

use crate::table::{DentTable, InsertResult, TableEntry, TrackingEntry};

pub use seize::{Guard, LocalGuard, OwnedGuard};

pub(crate) mod probe;
pub(crate) mod table;

pub struct JanusMap<K, V, S = RandomState> {
    base: DentTable<K, V, S>,
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

impl<K: Eq + Hash, V: Clone, S: BuildHasher + Clone> JanusMap<K, V, S> {
    fn new_inner(capacity: usize, hasher: S) -> JanusMap<K, V, S> {
        let base = DentTable::<K, V, S>::new(capacity, hasher);
        JanusMap { base }
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
        match self.base.insert(key, value, true, guard) {
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
        self.base.get(key, guard)
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
        self.base.get_mut(key, guard)
    }

    pub fn remove<Q>(&self, key: &Q, guard: &LocalGuard<'_>) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.base.remove(key, guard)
    }
}

impl<K, V, S> Drop for JanusMap<K, V, S> {
    fn drop(&mut self) {
        let table = self.base.inner.load(Ordering::Acquire);
        if table.is_null() {
            return;
        }
        DentTable::<K, V, S>::deallocate(table);
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
        &unsafe { &(*self.entry) }.target
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
    slot: *mut TableEntry<K, V>,
    _keep_alive: PhantomData<&'a ()>,
}

impl<'a, K, V> WriteGuard<'a, K, V> {
    fn as_ref(&self) -> &V {
        &unsafe { &(*self.write_ptr) }.target
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
        // self.slot is guaranteed to be non-null for as long as we have write access
        let old_read_ptr =
            unsafe { &(*self.slot) }.access[0].swap(self.write_ptr, Ordering::Release);
        unsafe { &(*self.slot) }.access[1].store(old_read_ptr, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

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

    #[test]
    fn insert_get_remove_test_70k() {
        let map = JanusMap::<u64, u64>::with_capacity(100_000);
        let guard = map.guard();

        for id in 0..70_000 {
            map.insert(id, id, &guard);
        }
        assert_eq!(map.len(), 70_000);

        for id in 0..70_000 {
            let read = map.get(&id, &guard).unwrap();
            assert_eq!(read.as_ref(), &id);
        }

        for id in 0..70_000 {
            let writer = map.get_mut(&id, &guard).unwrap();
            assert_eq!(writer.as_ref(), &id);
        }

        for id in 0..70_000 {
            let old = map.remove(&id, &guard).unwrap();
            assert_eq!(old, id);
        }

        assert_eq!(map.len(), 0);
    }

    #[test]
    fn insert_get_remove_test_50k() {
        let map = JanusMap::<u64, u64>::with_capacity(100_000);
        let guard = map.guard();

        for id in 0..50_000 {
            map.insert(id, id, &guard);
        }
        assert_eq!(map.len(), 50_000);

        for id in 0..50_000 {
            let read = map.get(&id, &guard).unwrap();
            assert_eq!(read.as_ref(), &id);
        }

        for id in 0..50_000 {
            let writer = map.get_mut(&id, &guard).unwrap();
            assert_eq!(writer.as_ref(), &id);
        }

        for id in 0..50_000 {
            let old = map.remove(&id, &guard).unwrap();
            assert_eq!(old, id);
        }

        assert_eq!(map.len(), 0);
    }

    #[test]
    fn insert_get_remove_test_20k() {
        let map = JanusMap::<u64, u64>::with_capacity(100_000);
        let guard = map.guard();

        for id in 0..20_000 {
            map.insert(id, id, &guard);
        }
        assert_eq!(map.len(), 20_000);

        for id in 0..20_000 {
            let read = map.get(&id, &guard).unwrap();
            assert_eq!(read.as_ref(), &id);
        }

        for id in 0..20_000 {
            let writer = map.get_mut(&id, &guard).unwrap();
            assert_eq!(writer.as_ref(), &id);
        }

        for id in 0..20_000 {
            let old = map.remove(&id, &guard).unwrap();
            assert_eq!(old, id);
        }

        assert_eq!(map.len(), 0);
    }

    #[test]
    fn insert_get_remove_test_80k_4_threads() {
        let map = Arc::new(JanusMap::<u64, u64>::with_capacity(100_000));

        let map1 = map.clone();
        let join1 = std::thread::spawn(move || {
            let guard = map1.guard();

            for id in 0..20_000 {
                map1.insert(id, id, &guard);
            }

            for id in 0..20_000 {
                let read = map1.get(&id, &guard).unwrap();
                assert_eq!(read.as_ref(), &id);
            }

            for id in 0..20_000 {
                let writer = map1.get_mut(&id, &guard).unwrap();
                assert_eq!(writer.as_ref(), &id);
            }

            for id in 0..20_000 {
                let old = map1.remove(&id, &guard).unwrap();
                assert_eq!(old, id);
            }
        });

        let map2 = map.clone();
        let join2 = std::thread::spawn(move || {
            let guard = map2.guard();

            for id in 0..20_000 {
                map2.insert(id * 10, id, &guard);
            }

            for id in 0..20_000 {
                let read = map2.get(&(id * 10), &guard).unwrap();
                assert_eq!(read.as_ref(), &id);
            }

            for id in 0..20_000 {
                let writer = map2.get_mut(&(id * 10), &guard).unwrap();
                assert_eq!(writer.as_ref(), &id);
            }

            for id in 0..20_000 {
                let old = map2.remove(&(id * 10), &guard).unwrap();
                assert_eq!(old, id);
            }
        });

        let map3 = map.clone();
        let join3 = std::thread::spawn(move || {
            let guard = map3.guard();

            for id in 0..20_000 {
                map3.insert(id * 100, id, &guard);
            }

            for id in 0..20_000 {
                let read = map3.get(&(id * 100), &guard).unwrap();
                assert_eq!(read.as_ref(), &id);
            }

            for id in 0..20_000 {
                let writer = map3.get_mut(&(id * 100), &guard).unwrap();
                assert_eq!(writer.as_ref(), &id);
            }

            for id in 0..20_000 {
                let old = map3.remove(&(id * 100), &guard).unwrap();
                assert_eq!(old, id);
            }
        });

        let map4 = map.clone();
        let join4 = std::thread::spawn(move || {
            let guard = map4.guard();

            for id in 0..20_000 {
                map4.insert(id * 1000, id, &guard);
            }

            for id in 0..20_000 {
                let read = map4.get(&(id * 1000), &guard).unwrap();
                assert_eq!(read.as_ref(), &id);
            }

            for id in 0..20_000 {
                let writer = map4.get_mut(&(id * 1000), &guard).unwrap();
                assert_eq!(writer.as_ref(), &id);
            }

            for id in 0..20_000 {
                let old = map4.remove(&(id * 1000), &guard).unwrap();
                assert_eq!(old, id);
            }
        });

        let _ = join1.join();
        let _ = join2.join();
        let _ = join3.join();
        let _ = join4.join();
    }
}
