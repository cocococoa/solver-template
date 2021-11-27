use solver_template::{Item, KSSolver};

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

fn main() {
    let mut items = vec![];

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
