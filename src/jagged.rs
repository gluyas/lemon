use std::default::Default;
use std::mem;
use std::ops::{Index, IndexMut};
use std::ptr;
use std::slice;

pub struct Jagged<T> {
    elems:   Vec<T>,
    indices: Vec<usize>,
}

impl<T> Jagged<T> {
    pub fn new() -> Self {
        Jagged { elems: vec![], indices: vec![0] }
    }

    pub fn len(&self) -> usize {
        // sub 1 to account for leading 0
        self.indices.len() - 1
    }

    pub fn total_len(&self) -> usize {
        self.elems.len()
    }

    pub fn push_boxed(&mut self, boxed: Box<[T]>) -> &mut [T] {
        unsafe {
            let push = self.push_raw(boxed.as_ref() as *const [T]);
            mem::transmute::<_, Box<mem::ManuallyDrop<[T]>>>(boxed);
            push
        }
    }

    pub fn push_vec(&mut self, vec: &mut Vec<T>) -> &mut [T] {
        unsafe {
            let push = self.push_raw(vec.as_ref() as *const [T]);
            vec.set_len(0);
            push
        }
    }

    pub unsafe fn push_raw(&mut self, src: *const [T]) -> &mut [T] {
        let len = (*src).len();
        let dst = self.push_uninit(len);
        ptr::copy_nonoverlapping(src as *const T, dst as *mut T, len);
        &mut *dst
    }

    #[must_use]
    pub unsafe fn push_uninit(&mut self, len: usize) -> *mut [T] {
        let old_index = self.elems.len();
        let new_index = old_index + len;
        self.indices.push(new_index);

        self.elems.reserve(len);
        self.elems.set_len(new_index);
        &mut self.elems[old_index..new_index] as *mut [T]
    }

    pub fn truncate(&mut self, len: usize) {
        if len < self.len() {
            self.elems.truncate(self.indices[len]);
            self.indices.truncate(len + 1);
        }
    }
}

impl<T> Jagged<T>
where
    T: Clone
{
    pub fn push_clone(&mut self, slice: &[T]) -> &mut [T] {
        let prev = *self.indices.last().expect("indices array missing leading 0");
        let next = prev + slice.len();
        self.indices.push(next);
        self.elems.extend_from_slice(slice);
        &mut self.elems[prev..next]
    }
}

impl<T> Jagged<T>
where
    T: Copy
{
    pub fn push_copy(&mut self, slice: &[T]) -> &mut [T] {
        unsafe { self.push_raw(slice as *const [T]) }
    }
}

impl<T> Jagged<T>
where
    T: Default
{
    pub fn push_default(&mut self, len: usize) -> &mut [T] {
        unsafe {
            let ref mut slice = *self.push_uninit(len);
            for elem in slice.iter_mut() {
                *elem = <T as Default>::default();
            }
            slice
        }
    }
}

impl<T> Index<usize> for Jagged<T> {
    type Output = [T];

    fn index(&self, idx: usize) -> &[T] {
        if idx >= self.len() { panic!("index out of range"); }
        &self.elems[self.indices[idx]..self.indices[idx+1]]
    }
}

impl<T> IndexMut<usize> for Jagged<T> {
    fn index_mut(&mut self, idx: usize) -> &mut [T] {
        if idx >= self.len() { panic!("index out of range"); }
        &mut self.elems[self.indices[idx]..self.indices[idx+1]]
    }
}
