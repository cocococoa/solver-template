use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter};
use std::time::{Duration, Instant};

#[derive(Debug, Eq, PartialEq)]
pub struct Item {
    name: String,
    value: u64,
    weight: u64,
}
impl PartialOrd for Item {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        // comparet self.eff()
        let a = self.value * other.weight;
        let b = self.weight * other.value;

        match a.cmp(&b) {
            Ordering::Equal => self.value.cmp(&other.value),
            Ordering::Greater => Ordering::Greater,
            Ordering::Less => Ordering::Less,
        }
    }
}
impl Display for Item {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{}: value: {}, weight: {}",
            self.name,
            self.value,
            self.weight()
        )
    }
}
impl Item {
    pub fn new(name: String, value: u64, weight: u64) -> Item {
        Item {
            name: name,
            value: value,
            weight: weight,
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn value(&self) -> u64 {
        self.value
    }
    pub fn weight(&self) -> u64 {
        self.weight
    }
    pub fn eff(&self) -> f64 {
        self.value as f64 / self.weight as f64
    }
}

#[derive(Debug)]
struct KSDebug {
    n: usize,
    searched_node: u32,
    searched_leaf: u32,
    pruned_leaf: HashMap<usize, u32>,
}
impl KSDebug {
    fn new(n: usize) -> Self {
        KSDebug {
            n: n,
            searched_node: 0,
            searched_leaf: 0,
            pruned_leaf: HashMap::new(),
        }
    }
    fn visit_node(&mut self) {
        self.searched_node += 1;
    }
    fn visit_leaf(&mut self) {
        self.searched_leaf += 1;
    }
    fn prune(&mut self, ord: usize) {
        match self.pruned_leaf.get_mut(&ord) {
            Some(value) => {
                *value += 1;
            }
            None => {
                self.pruned_leaf.insert(ord, 1);
            }
        }
    }
    fn calc_prune_leaf(&self) -> u128 {
        let mut ret = 0;
        for (ord, num) in self.pruned_leaf.iter() {
            ret += *num as u128 * (1 << *ord as u128);
        }
        ret
    }
    fn estimate_progress(&self) -> f64 {
        let mut v = vec![0; self.n + 1];
        for (ord, num) in self.pruned_leaf.iter() {
            v[*ord] = *num;
        }
        v[0] += self.searched_leaf;
        let mut remain = 0;
        for x in v.iter_mut() {
            *x += remain;
            remain = *x / 2;
            *x = *x % 2;
        }
        if v[self.n] == 1 {
            return 1.0;
        }

        let mut all_one = true;
        let mut ret = 0.0;
        for i in 1..30 {
            if self.n >= i {
                if v[self.n - i] == 1 {
                    ret += 1.0 / (1 << i) as f64;
                } else {
                    all_one = false;
                }
            }
        }

        if all_one {
            1.0
        } else {
            ret
        }
    }
}

#[derive(Debug)]
pub struct KSSolver {
    items: Vec<Item>,
    timeout: Duration,
    max_weight: u64,
}
impl KSSolver {
    pub fn new(mut items: Vec<Item>, max_weight: u64) -> KSSolver {
        items.sort_by(|a, b| b.cmp(a));
        let mut dict = HashMap::new();
        for (i, item) in items.iter().enumerate() {
            dict.insert(item.name(), i);
        }

        KSSolver {
            items: items,
            timeout: Duration::from_secs(1),
            max_weight: max_weight,
        }
    }
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }
    pub fn timeout(&self) -> Duration {
        self.timeout
    }
    pub fn set_max_weight(&mut self, max_weight: u64) {
        self.max_weight = max_weight;
    }
    pub fn max_weight(&self) -> u64 {
        self.max_weight
    }
    pub fn search(&self, i: &str) -> Option<&Item> {
        self.items.iter().find(|x| x.name == i)
    }
    pub fn run(&self) -> KSResult {
        let start = Instant::now();
        let n = self.items.len();

        // for debug
        let mut ks_debug = KSDebug::new(n);

        // for return
        let mut known_best = vec![];
        let mut lb = 0;
        let mut weight_sum = 0;

        // for search
        let mut stack = vec![];
        if n > 0 {
            let first_item = self.items.get(0).unwrap();
            stack.push(SubProblem::new(0, false, 0, 0));
            if first_item.weight() <= self.max_weight {
                stack.push(SubProblem::new(
                    0,
                    true,
                    first_item.value(),
                    first_item.weight(),
                ));
            } else {
                ks_debug.prune(n - 1);
            }
        }
        let mut cur_state = vec![];
        cur_state.reserve(n);

        while !stack.is_empty() {
            ks_debug.visit_node();

            let parent = stack.pop().unwrap();

            // set cur_state to the state of parent
            while cur_state.len() > parent.depth {
                cur_state.pop();
            }
            if parent.select {
                cur_state.push(true);
            } else {
                cur_state.push(false);
            }

            // if reached to the leaf
            if parent.depth + 1 == n {
                ks_debug.visit_leaf();

                // update known best
                if parent.value_sum > lb {
                    known_best = cur_state.clone();
                    lb = parent.value_sum;
                    weight_sum = parent.weight_sum;
                }
                continue;
            }

            // branch and bound
            let children = branch(parent, &self.items, self.max_weight, &mut ks_debug);
            for child in children.into_iter().rev() {
                let ub = bound(&child, &self.items, self.max_weight);

                // if the upper bound of child is smaller than lb,
                // we prune this subproblem.
                if ub <= lb {
                    ks_debug.prune(n - child.depth - 1);
                    continue;
                }

                stack.push(child);
            }

            if start.elapsed() > self.timeout {
                break;
            }
        }

        // for debug
        println!("Elapsed time : {} [us]", start.elapsed().as_micros());
        println!("Progress     : {} %", 100.0 * ks_debug.estimate_progress());
        println!("Searched node: {}", ks_debug.searched_node);
        println!("Searched leaf: {}", ks_debug.searched_leaf);
        if n <= 127 {
            println!("Pruned leaf  : {}", ks_debug.calc_prune_leaf());
            debug_assert_eq!(
                1 << n,
                ks_debug.searched_leaf as u128 + ks_debug.calc_prune_leaf()
            );
        }

        // calculate upper bound
        {
            let mut ub = 0.0;
            let mut remain_weight = self.max_weight();
            for i in 0..n {
                let x = self.items.get(i).unwrap();
                if x.weight() <= remain_weight {
                    ub += x.value() as f64;
                    remain_weight -= x.weight();
                } else {
                    ub += x.eff() * remain_weight as f64;
                    break;
                }
            }
            println!("upper bound  : {}", ub as u64);
        }

        let mut set = HashSet::new();
        for i in 0..known_best.len() {
            if known_best[i] {
                set.insert(self.items[i].name());
            }
        }
        KSResult {
            known_best: set,
            value_sum: lb,
            weight_sum: weight_sum,
            is_best: stack.is_empty(),
            elapsed: start.elapsed(),
        }
    }
}

#[derive(Debug)]
pub struct KSResult<'a> {
    known_best: HashSet<&'a str>,
    value_sum: u64,
    weight_sum: u64,
    is_best: bool,
    elapsed: Duration,
}
impl KSResult<'_> {
    pub fn get(&self, name: &str) -> bool {
        self.known_best.contains(name)
    }
    pub fn list(&self) -> Vec<&str> {
        self.known_best.iter().map(|x| *x).collect::<Vec<_>>()
    }
    pub fn value_sum(&self) -> u64 {
        self.value_sum
    }
    pub fn weight_sum(&self) -> u64 {
        self.weight_sum
    }
    pub fn is_best(&self) -> bool {
        self.is_best
    }
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }
}

struct SubProblem {
    depth: usize,
    select: bool,
    value_sum: u64,
    weight_sum: u64,
}
impl SubProblem {
    fn new(depth: usize, select: bool, value_sum: u64, weight_sum: u64) -> SubProblem {
        SubProblem {
            depth: depth,
            select: select,
            value_sum: value_sum,
            weight_sum: weight_sum,
        }
    }
}

// create subproblems
fn branch(
    parent: SubProblem,
    items: &Vec<Item>,
    max_weight: u64,
    ks_debug: &mut KSDebug,
) -> Vec<SubProblem> {
    let child_depth = parent.depth + 1;
    debug_assert!(child_depth < items.len());

    let item = items.get(child_depth).unwrap();
    let child_out = SubProblem {
        depth: child_depth,
        select: false,
        value_sum: parent.value_sum,
        weight_sum: parent.weight_sum,
    };

    if parent.weight_sum + item.weight() <= max_weight {
        let child_in = SubProblem {
            depth: child_depth,
            select: true,
            value_sum: parent.value_sum + item.value(),
            weight_sum: parent.weight_sum + item.weight(),
        };
        vec![child_in, child_out]
    } else {
        ks_debug.prune(items.len() - child_depth - 1);
        vec![child_out]
    }
}
// calculate upper bound of subproblem
fn bound(sp: &SubProblem, items: &Vec<Item>, max_weight: u64) -> u64 {
    let mut remain_weight = max_weight - sp.weight_sum;
    let mut max_value = sp.value_sum;

    // TODO: implement without loop
    for i in (sp.depth + 1)..items.len() {
        let item = items.get(i).unwrap();
        if item.weight() <= remain_weight {
            remain_weight -= item.weight();
            max_value += item.value();
        } else {
            max_value += (item.eff() * remain_weight as f64) as u64;
            break;
        }
    }

    max_value as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item() {
        let a = Item::new("a".to_string(), 30, 15);
        let b = Item::new("b".to_string(), 3, 2);

        assert_eq!(2 as f64, a.eff());
        assert_eq!("a", a.name());
        assert_eq!("a: value: 30, weight: 15", format!("{}", a));
        assert_eq!(1.5 as f64, b.eff());
        assert_eq!("b", b.name());
        assert_eq!("b: value: 3, weight: 2", format!("{}", b));
        assert!(a > b);
    }
    #[test]
    fn test_solver_small() {
        // ref: https://qiita.com/drken/items/a5e6fe22863b7992efdb
        let items = vec![
            Item::new("a".to_string(), 3, 2),
            Item::new("b".to_string(), 2, 1),
            Item::new("c".to_string(), 6, 3),
            Item::new("d".to_string(), 1, 2),
            Item::new("e".to_string(), 3, 1),
            Item::new("f".to_string(), 85, 5),
        ];

        let solver = KSSolver::new(items, 9);
        let result = solver.run();

        println!("Solution: {:?}", result.list());

        assert!(result.get("c"));
        assert!(result.get("e"));
        assert!(result.get("f"));
        assert!(!result.get("a"));
        assert!(!result.get("b"));
        assert!(!result.get("d"));
        assert_eq!(94, result.value_sum());
        assert!(result.is_best());
    }

    #[test]
    fn test_solver_large() {
        // ref: https://atcoder.jp/contests/abc032/tasks/abc032_d
        let items = vec![
            Item::new("a".to_string(), 128990795, 137274936),
            Item::new("b".to_string(), 575374246, 989051853),
            Item::new("c".to_string(), 471048785, 85168425),
            Item::new("d".to_string(), 640066776, 856699603),
            Item::new("e".to_string(), 819841327, 611065509),
            Item::new("f".to_string(), 704171581, 22345022),
            Item::new("g".to_string(), 536108301, 678298936),
            Item::new("h".to_string(), 119980848, 616908153),
            Item::new("i".to_string(), 117241527, 28801762),
            Item::new("j".to_string(), 325850062, 478675378),
            Item::new("k".to_string(), 623319578, 706900574),
            Item::new("l".to_string(), 998395208, 738510039),
            Item::new("m".to_string(), 475707585, 135746508),
            Item::new("n".to_string(), 863910036, 599020879),
            Item::new("o".to_string(), 340559411, 738084616),
            Item::new("p".to_string(), 122579234, 545330137),
            Item::new("q".to_string(), 696368935, 86797589),
            Item::new("r".to_string(), 665665204, 592749599),
            Item::new("s".to_string(), 958833732, 401229830),
            Item::new("t".to_string(), 371084424, 523386474),
            Item::new("u".to_string(), 463433600, 5310725),
            Item::new("v".to_string(), 210508742, 907821957),
            Item::new("w".to_string(), 685281136, 565237085),
            Item::new("z".to_string(), 619500108, 730556272),
            Item::new("y".to_string(), 88215377, 310581512),
            Item::new("z".to_string(), 558193168, 136966252),
            Item::new("1".to_string(), 475268130, 132739489),
            Item::new("2".to_string(), 303022740, 12425915),
            Item::new("3".to_string(), 122379996, 137199296),
            Item::new("4".to_string(), 304092766, 23505143),
        ];

        let solver = KSSolver::new(items, 499887702);
        let result = solver.run();

        println!("Solution: {:?}", result.list());

        assert_eq!(3673016420, result.value_sum());
        assert!(result.is_best());
    }
}
