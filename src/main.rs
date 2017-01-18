extern crate arrayfire;
extern crate csv;

use arrayfire::*;
use std::f32;

use std::fs::File;
use std::io::{Read, Write};

use std::env;

fn save_matrix(f: &str, m: &Array) {
    let m = transpose(m, false);
    let mut v = Vec::new();
    v.resize(m.elements() as usize, 0f32);
    m.host(&mut v);
    let dim = m.dims();
    let mut f = File::create(f).unwrap();
    writeln!(f, "{} {}", dim[1], dim[0]).unwrap();
    for i in 0 .. dim[1] {
        for j in 0 .. dim[0] {
            write!(f, "{} ", v[(dim[0] * i + j) as usize]).unwrap();
        }
        writeln!(f, "").unwrap();
    }
}

fn read_matrix(f: &str) -> Array {
    let mut f = File::open(f).unwrap();
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();

    let mut rdr = csv::Reader::from_string(s).has_headers(true).flexible(true).delimiter(b' ');
    let elements = rdr.decode().collect::<csv::Result<Vec<Vec<f32>>>>().unwrap()
        .into_iter()
        .flat_map(|x|x.into_iter())
        .collect::<Vec<f32>>();
    let dim = rdr.headers().unwrap().iter()
        .map(|x| x.parse::<u64>())
        .filter(|x|x.is_ok()).map(|x| x.unwrap()).collect::<Vec<u64>>();
    assert_eq!(dim.len(), 2);
    assert_eq!(elements.len() as u64, dim[0] * dim[1]);

    transpose(&Array::new(&elements, Dim4::new(&[dim[1], dim[0], 1, 1])), false)
}

fn main() {
    let args = env::args().collect::<Vec<String>>();
    if args.len() != 5 {
        println!("Usage: {} file1 iter_time max_error dim2",
            args[0]);
        return;
    }
    let input_file = &args[1];
    let max_iter = args[2].parse::<usize>().unwrap_or(100);
    let max_error = args[3].parse::<f32>().unwrap_or(0.001);
    let target_dim = args[4].parse::<u64>().unwrap();
    println!("Parameters: file1={}, iter={}, error={}, ndim={}",
        input_file, max_iter, max_error, target_dim);

    let backends = get_available_backends();
    if backends.contains(&Backend::CUDA) {
        set_backend(Backend::CUDA);
        println!("Use {} CUDAs", device_count());
    } else if backends.contains(&Backend::OPENCL) {
        set_backend(Backend::OPENCL);
        println!("Use {} OPENCLs", device_count());
    } else {
        set_backend(Backend::CPU);
        println!("Use {} CPU", device_count());
    }

    info();

    // Non-negative Matrix Factorization
    // A = WH
    let A = read_matrix(input_file);
    let mut error = f32::INFINITY;
    let mut old_error = f32::INFINITY;
    let mut i = 0;

    let input_file_dimension = A.dims();
    let mut w = randu::<f32>(Dim4::new(&[input_file_dimension[0], target_dim, 1, 1]));
    let mut h = randu::<f32>(Dim4::new(&[target_dim, input_file_dimension[1], 1, 1]));

    while error > max_error && i < max_iter {
        w = mul(&w, &div(
            &matmul(&A, &w, MatProp::NONE, MatProp::TRANS),
            &matmul(&matmul(&w, &h, MatProp::NONE, MatProp::NONE), &h, MatProp::NONE, MatProp::TRANS)
        ,false), false);
        h = mul(&h, &div(
            &matmul(&w, &A, MatProp::TRANS, MatProp::NONE),
            &matmul(&w, &matmul(&w, &h, MatProp::NONE, MatProp::NONE), MatProp::TRANS, MatProp::NONE)
        ,false), false);
        // calculate the sequared-error
        let e1 = sub(&A, &matmul(&w, &h, MatProp::NONE, MatProp::NONE), false);
        let e1 = sum_all(&mul(&e1, &e1, false)).0 as f32;
        error = (old_error - e1).abs();
        old_error = e1;
        i += 1;
        println!("iter:{:05} current error:{:.7}, step:{:7}", i, e1, error);
    }
    save_matrix("w.mat", &w);
    save_matrix("h.mat", &h);
}
