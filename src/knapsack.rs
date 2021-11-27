use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter};
use std::time::{Duration, Instant};

#[derive(Debug, Eq, PartialEq)]
pub struct Item<'a> {
    name: &'a str,
    value: u64,
    weight: u64,
}
impl PartialOrd for Item<'_> {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
impl Ord for Item<'_> {
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
impl Display for Item<'_> {
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
impl Item<'_> {
    pub fn new<'a>(name: &'a str, value: u64, weight: u64) -> Item<'a> {
        Item {
            name: name,
            value: value,
            weight: weight,
        }
    }
    pub fn name<'a>(&'a self) -> &'a str {
        self.name
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
pub struct KSSolver<'a> {
    items: Vec<Item<'a>>,
    timeout: Duration,
    max_weight: u64,
    dict: HashMap<&'a str, usize>,
}
impl KSSolver<'_> {
    pub fn new(mut items: Vec<Item>, max_weight: u64) -> KSSolver {
        items.sort_by(|a, b| b.cmp(a));
        let mut dict = HashMap::new();
        for (i, item) in items.iter().enumerate() {
            dict.insert(item.name, i);
        }

        KSSolver {
            items: items,
            timeout: Duration::from_secs(1),
            max_weight: max_weight,
            dict: dict,
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
    pub fn get(&self, i: &str) -> Option<&Item> {
        match self.dict.get(i) {
            Some(index) => self.items.get(*index),
            None => None,
        }
    }
    pub fn run(&self) -> KSResult {
        let start = Instant::now();

        // for return
        let mut known_best = vec![];
        let mut lb = 0;

        // for search
        let mut stack = vec![];
        if self.items.len() > 0 {
            let first_item = self.items.get(0).unwrap();
            stack.push(SubProblem::new(0, false, 0, 0));
            if first_item.weight() <= self.max_weight {
                stack.push(SubProblem::new(
                    0,
                    true,
                    first_item.value(),
                    first_item.weight(),
                ));
            }
        }
        let mut cur_state = vec![];
        cur_state.reserve(self.items.len());

        // for debug
        let mut bounded: u64 = 0;

        while !stack.is_empty() {
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
            if parent.depth + 1 == self.items.len() {
                // update known best
                if parent.value_sum > lb {
                    lb = parent.value_sum;
                    known_best = cur_state.clone();
                }
                continue;
            }

            // branch and bound
            let children = branch(parent, &self.items, self.max_weight);
            for child in children.into_iter().rev() {
                let ub = bound(&child, &self.items, self.max_weight);

                // if the upper bound of child is smaller than lb,
                // we prune this subproblem.
                if ub <= lb {
                    bounded += 1 << (self.items.len() - child.depth - 1);
                    continue;
                }

                stack.push(child);
            }

            if start.elapsed() > self.timeout {
                break;
            }
        }

        // for debug
        println!("Bounded: {}", bounded);

        let mut set = HashSet::new();
        for i in 0..known_best.len() {
            if known_best[i] {
                set.insert(self.items[i].name);
            }
        }
        KSResult {
            known_best: set,
            value_sum: lb,
            is_best: stack.is_empty(),
        }
    }
}

#[derive(Debug)]
pub struct KSResult<'a> {
    known_best: HashSet<&'a str>,
    value_sum: u64,
    is_best: bool,
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
    pub fn is_best(&self) -> bool {
        self.is_best
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
fn branch(parent: SubProblem, items: &Vec<Item>, max_weight: u64) -> Vec<SubProblem> {
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
        let a = Item::new("a", 30, 15);
        let b = Item::new("b", 3, 2);

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
            Item::new("a", 3, 2),
            Item::new("b", 2, 1),
            Item::new("c", 6, 3),
            Item::new("d", 1, 2),
            Item::new("e", 3, 1),
            Item::new("f", 85, 5),
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
            Item::new("a", 128990795, 137274936),
            Item::new("b", 575374246, 989051853),
            Item::new("c", 471048785, 85168425),
            Item::new("d", 640066776, 856699603),
            Item::new("e", 819841327, 611065509),
            Item::new("f", 704171581, 22345022),
            Item::new("g", 536108301, 678298936),
            Item::new("h", 119980848, 616908153),
            Item::new("i", 117241527, 28801762),
            Item::new("j", 325850062, 478675378),
            Item::new("k", 623319578, 706900574),
            Item::new("l", 998395208, 738510039),
            Item::new("m", 475707585, 135746508),
            Item::new("n", 863910036, 599020879),
            Item::new("o", 340559411, 738084616),
            Item::new("p", 122579234, 545330137),
            Item::new("q", 696368935, 86797589),
            Item::new("r", 665665204, 592749599),
            Item::new("s", 958833732, 401229830),
            Item::new("t", 371084424, 523386474),
            Item::new("u", 463433600, 5310725),
            Item::new("v", 210508742, 907821957),
            Item::new("w", 685281136, 565237085),
            Item::new("z", 619500108, 730556272),
            Item::new("y", 88215377, 310581512),
            Item::new("z", 558193168, 136966252),
            Item::new("1", 475268130, 132739489),
            Item::new("2", 303022740, 12425915),
            Item::new("3", 122379996, 137199296),
            Item::new("4", 304092766, 23505143),
        ];

        let solver = KSSolver::new(items, 499887702);
        let result = solver.run();

        println!("Solution: {:?}", result.list());

        assert_eq!(3673016420, result.value_sum());
        assert!(result.is_best());
    }
}
