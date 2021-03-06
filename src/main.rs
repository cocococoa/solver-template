extern crate getopts;
use getopts::Options;
use solver_template::{Item, KSSolver};
use std::env;

fn read<T>(line: &str) -> (T, T)
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let mut x = line
        .split_whitespace()
        .map(|x| x.parse::<T>().unwrap())
        .collect::<Vec<_>>();
    let second = x.pop().unwrap();
    let first = x.pop().unwrap();
    (first, second)
}

fn solve_knapsack() {
    println!("===================================");
    println!("Solve knapsack problem");
    println!("Input format:");
    println!("    N   W");
    println!("    v_1 w_1");
    println!("    v_2 w_2");
    println!("    ...");
    println!("    v_N w_N");
    let mut items = vec![];

    // FIXME: too late
    use std::io::{self, BufRead};
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut line = String::new();
    input.read_line(&mut line).unwrap();
    let (n, w) = read(&line);
    for i in 0..n {
        input.read_line(&mut line).unwrap();
        let (v, w) = read::<u64>(&line);
        items.push(Item::new(format!("{}", i + 1), v, w));
    }

    println!("===================================");
    println!("Run knapsack solver");
    let solver = KSSolver::new(items, w as u64);
    let result = solver.run();

    println!("===================================");
    if result.is_best() {
        println!("Best solution has found");
    } else {
        println!("Best solution has not found");
    }
    println!("Maximum: {}", result.value_sum());
    println!("Weight: {}", result.weight_sum());
    let mut list = result.list();
    list.sort_by(|a, b| {
        let na = a.parse::<usize>().unwrap();
        let nb = b.parse::<usize>().unwrap();
        na.cmp(&nb)
    });
    for i in list {
        println!("Select {}", solver.search(i).unwrap());
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut opts = Options::new();
    opts.reqopt("t", "type", "Type of problem to solve", "");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => panic!("{}", f),
    };

    let _knapsack = "knapsack".to_string();
    match matches.opt_str("t") {
        Some(_knapsack) => solve_knapsack(),
        _ => unimplemented!(),
    }
}
