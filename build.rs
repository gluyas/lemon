extern crate gl_generator;

use gl_generator::{Registry, Api, Profile, Fallbacks, GlobalGenerator};
use std::env;
use std::fs::{self, File};
use std::path::Path;

fn main() {
    print!("writing GL bindings... ");
    let dir = env::var("OUT_DIR").unwrap();
    let mut gl_bindings = File::create(&Path::new(&dir).join("gl_bindings.rs")).unwrap();

    Registry::new(Api::Gles2, (3, 0), Profile::Core, Fallbacks::All, [])
        .write_bindings(GlobalGenerator, &mut gl_bindings)
        .unwrap();
    println!("done.");

    for file in fs::read_dir("./src/static") .unwrap() {
        let file = file.unwrap();
        fs::copy(file.path(), &Path::new(&dir).join("../../../").join(file.file_name())).unwrap();
    }
}
