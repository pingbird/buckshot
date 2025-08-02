#![allow(dead_code)]
#![allow(unused_imports)]

use crate::state::{
    fmt_percent, fmt_weight, gen_lut, min_cover, Action, BitArray, Game, GameConfig, Gosper, Item,
    ItemAction, ItemMap, KnownChance, Player, EPSILON,
};
use slab::Slab;
use smallvec::{smallvec, SmallVec};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::ops::Mul;

mod state;

fn main() {
    // let lut = gen_lut();
    // for i in lut[2].iter() {
    //     println!("{:0>8b}", i);
    // }
    // std::process::exit(0);

    let game: Game<false, 2> = Game {
        max_health: 2,
        players: [
            Player {
                health: 2,
                items: ItemMap::new()
                .with(Item::Adrenaline, 1)
                .with(Item::HandSaw, 1),
                known_shells: BitArray(0b0),
            },
            Player {
                health: 2,
                items: ItemMap::new(),
                known_shells: BitArray(0b0),
            },
        ],
        handcuffed: BitArray(0b00),
        live_shells: BitArray(0b1),
        n_shells: 1,
        turn: 0,
        sawed_off: false,
        forward: true,
    };

    let mut tree = Tree::new(&game);
    tree.get_decisions(tree.root, 5);
    // Run mmdc -i output.mmd -o output.png and then open output.png in chrome
    let (s, n_links) = tree.to_mermaid(Some(0));
    std::fs::write("output.mmd", s).unwrap();
    let output = std::process::Command::new(r"C:\Users\ping\AppData\Roaming\npm\mmdc.cmd")
        .args([
            "-i",
            "output.mmd",
            "--configFile",
            "mermaidRenderConfig.json",
            "-o",
            "output.png",
            "-t",
            "dark",
            "-b",
            "#0e0e0e",
            "-s",

            &n_links.div_ceil(25).clamp(1, 20).to_string(),
        ])
        .output()
        .unwrap();
    if !output.status.success() {
        eprintln!("mmdc failed: {}", String::from_utf8_lossy(&output.stderr));
        std::process::exit(1);
    }
    std::process::Command::new("explorer")
        .arg("output.png")
        .output()
        .unwrap();
    // println!("{}", game);
    //
    // let mut actions = vec![];
    // game.visit_certain(&mut actions, &mut |actions, game| {
    //     println!(
    //         "{}",
    //         actions
    //             .iter()
    //             .map(|e| format!("{}", e))
    //             .collect::<Vec<_>>()
    //             .join(" -> ")
    //     );
    //     // println!("{}", game);
    //     println!("{}", "=".repeat(10));
    // });
}

fn permute_shells(n_shells: u8, live: u8) -> Vec<BitArray<u8>> {
    let mut shells = Vec::new();
    for i in 0..=(1u8 << n_shells).wrapping_sub(1) {
        if i.count_ones() == live.count_ones() {
            shells.push(BitArray(i));
        }
    }
    shells
}

struct Outcome<const PLAYERS: usize> {
    chance: [f32; PLAYERS],
    node: usize,
}

struct Permutation<const PLAYERS: usize> {
    reveal: Option<(u8, bool, Option<u8>)>,
    outcomes: SmallVec<[Outcome<PLAYERS>; 2]>,
}

struct Decision<const PLAYERS: usize> {
    actions: SmallVec<[Action; 4]>,
    weights: [f32; PLAYERS],
    permutations: SmallVec<[Permutation<PLAYERS>; 2]>,
}

struct Decisions<const PLAYERS: usize> {
    decisions: SmallVec<[Decision<PLAYERS>; 2]>,
    depth: usize,
    best: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key<const MULTIPLAYER: bool, const PLAYERS: usize> {
    pub game: Game<MULTIPLAYER, PLAYERS>,
}

struct Node<const MULTIPLAYER: bool, const PLAYERS: usize> {
    pub key: Key<MULTIPLAYER, PLAYERS>,
    pub decisions: Option<Decisions<PLAYERS>>,
}

struct Tree<const MULTIPLAYER: bool, const PLAYERS: usize> {
    nodes: Slab<Node<MULTIPLAYER, PLAYERS>>,
    root: usize,
    nodes_by_key: HashMap<Key<MULTIPLAYER, PLAYERS>, usize>,
}

impl<const MULTIPLAYER: bool, const PLAYERS: usize> Node<MULTIPLAYER, PLAYERS> {
    fn game(&self) -> &Game<MULTIPLAYER, PLAYERS> {
        &self.key.game
    }

    fn player_points(game: &Game<MULTIPLAYER, PLAYERS>, player_index: u8) -> i32 {
        let mut points: i32 = 0;
        let player = &game.players[player_index as usize];
        if player.health == 0 {
            return 0;
        }
        points += (player.health as i32) * 16;
        let mut n_items = 0;
        for item in Item::ALL {
            match item {
                Item::MagnifyingGlass => n_items += *player.items.get(item) as i32 * 2,
                Item::Adrenaline | Item::Handcuffs => n_items += *player.items.get(item) as i32 * 4,
                _ => n_items += *player.items.get(item) as i32 * 1,
            }
        }
        if game.n_shells == 0 {
            n_items = (n_items + 3).min(8);
        }
        points += n_items * 1;
        points += player.known_shells.count() as i32;
        if game.handcuffed.get(player_index) && !game.live_shells.empty() {
            points = points.saturating_sub(6);
        }
        points
    }

    fn all_player_points(game: &Game<MULTIPLAYER, PLAYERS>) -> ([i32; PLAYERS], i32) {
        let mut points = [0; PLAYERS];
        let mut total = 0;
        for i in 0..PLAYERS {
            let player_points = Self::player_points(game, i as u8);
            points[i] = player_points;
            total += player_points;
        }
        (points, total)
    }

    fn heuristic_weights(game: &Game<MULTIPLAYER, PLAYERS>) -> [f32; PLAYERS] {
        let (points, total) = Self::all_player_points(game);
        let mut weights = [0.0; PLAYERS];
        for i in 0..PLAYERS {
            weights[i] = points[i] as f32 / total as f32;
        }
        weights
    }
}

impl<const MULTIPLAYER: bool, const PLAYERS: usize> Tree<MULTIPLAYER, PLAYERS> {
    pub fn new(base_game: &Game<MULTIPLAYER, PLAYERS>) -> Self {
        let base_game = base_game.clone().normalize_known();
        let mut nodes = Slab::new();
        let key = Key { game: base_game };
        let root = nodes.insert(Node {
            key: key.clone(),
            decisions: None,
        });
        let mut nodes_by_key = HashMap::new();
        nodes_by_key.insert(key, root);
        Tree {
            nodes,
            root,
            nodes_by_key,
        }
    }

    fn visit_shell_configurations<F: FnMut(BitArray<u8>)>(
        n_shells: u8,
        live: u8,
        known: u8,
        f: &mut F,
    ) {
        let known_live = live & known;
        for i in 0..=((1u32 << n_shells) - 1) as u8 {
            if i & known == known_live && i.count_ones() == live.count_ones() {
                f(BitArray(i));
            }
        }
    }

    pub fn add_game(&mut self, key: Key<MULTIPLAYER, PLAYERS>) -> usize {
        if let Some(node) = false.then(|| self.nodes_by_key.get(&key)).flatten() {
            *node
        } else {
            let node = Node {
                key: key.clone(),
                decisions: None,
            };
            let node_index = self.nodes.insert(node);
            self.nodes_by_key.insert(key, node_index);
            node_index
        }
    }

    fn apply_outcome(
        &mut self,
        game: Game<MULTIPLAYER, PLAYERS>,
        weights: &mut [f32; PLAYERS],
        max_depth: usize,
        chance: [f32; PLAYERS],
    ) -> Outcome<PLAYERS> {
        let key = Key { game: game.clone() };
        let outcome = self.add_game(key);
        if max_depth == 0 || game.round_over() {
            // Use heuristics
            let heuristic_weights = Node::heuristic_weights(&self.nodes[outcome].key.game);
            for i in 0..PLAYERS {
                weights[i] += heuristic_weights[i] * chance[i];
            }
        } else {
            let outcome_decisions = self.get_decisions(outcome, max_depth - 1);
            match outcome_decisions.best {
                None => {
                    // Game over, winner takes all
                    let winner = self.nodes[outcome].key.game.turn;
                    weights[winner as usize] += chance[winner as usize];
                }
                Some(best) => {
                    // Assume the best decision will be taken
                    let best_decision = &outcome_decisions.decisions[best];
                    for j in 0..PLAYERS {
                        weights[j] += best_decision.weights[j] * chance[j];
                    }
                }
            }
        }
        Outcome {
            chance,
            node: outcome,
        }
    }

    fn apply_action(
        &mut self,
        key: Key<MULTIPLAYER, PLAYERS>,
        action: Option<Action>,
        weights: &mut [f32; PLAYERS],
        max_depth: usize,
        chance: [f32; PLAYERS],
    ) -> SmallVec<[Outcome<PLAYERS>; 2]> {
        if chance == [0.0; PLAYERS] {
            return smallvec![];
        }
        if let Some(action) = action {
            let mut outcomes = smallvec![];
            key.game.apply_action(action, |game, outcome_chance| {
                outcomes.push(self.apply_outcome(
                    game,
                    weights,
                    max_depth,
                    chance.map(|c| c * outcome_chance),
                ))
            });
            outcomes
        } else {
            smallvec![self.apply_outcome(key.game, weights, max_depth, chance)]
        }
    }

    pub fn compute_decisions(&mut self, node: usize, max_depth: usize) -> Decisions<PLAYERS> {
        let game = self.nodes[node].game().clone();
        let turn = game.turn;
        let mut decisions = SmallVec::new();
        let mut best_weight = f32::MIN;
        let mut best = None;
        let mut seen: SmallVec<[(Game<MULTIPLAYER, PLAYERS>, Option<Action>); 8]> = smallvec![];
        game.visit_certain(&mut |actions: &[Action],
                                 game: &Game<MULTIPLAYER, PLAYERS>,
                                 uncertain_action| {
            eprintln!("visit_certain actions: {:?} uncertain_action: {:?}", actions, uncertain_action);
            let game = game.clone();
            if seen.contains(&(game.clone(), uncertain_action)) {
                return;
            }
            seen.push((game.clone(), uncertain_action));
            let mut weights = [0.0; PLAYERS];
            let mut permutations: SmallVec<[Permutation<PLAYERS>; 2]> = smallvec![];

            let mut actions: SmallVec<[Action; 4]> = actions.iter().copied().collect();

            if let Some(action) = uncertain_action {
                let original_live = game.live_shells;
                game.visit_action_perspective(action, |permutation| {
                    let mut outcomes = smallvec![];
                    for (game, chance) in permutation.outcomes.into_iter() {
                        outcomes.push(self.apply_outcome(game, &mut weights, max_depth, chance));
                    }
                    permutations.push(Permutation {
                        reveal: permutation.reveal.map(|(shell, swap)| {
                            (shell, original_live.get(shell) != swap.is_some(), swap)
                        }),
                        outcomes,
                    });
                });
                actions.push(action);
            } else {
                permutations.push(Permutation {
                    reveal: None,
                    outcomes: smallvec![self.apply_outcome(
                        game.clone(),
                        &mut weights,
                        max_depth,
                        [1.0; PLAYERS]
                    )],
                });
            }

            let weight = weights[turn as usize];
            if weight > best_weight {
                best_weight = weight;
                best = Some(decisions.len());
            }

            decisions.push(Decision {
                actions,
                weights,
                permutations,
            });
        });
        Decisions {
            decisions,
            depth: max_depth,
            best,
        }
    }

    fn get_decisions(&mut self, node: usize, max_depth: usize) -> &mut Decisions<PLAYERS> {
        if self.nodes[node].decisions.is_some() {
            let decisions = self.nodes[node].decisions.as_ref().unwrap();
            if decisions.depth <= max_depth {
                let decisions = self.compute_decisions(node, max_depth);
                self.nodes[node].decisions = Some(decisions);
            }
        } else {
            let decisions = self.compute_decisions(node, max_depth);
            self.nodes[node].decisions = Some(decisions);
        }
        self.nodes[node].decisions.as_mut().unwrap()
    }

    pub fn to_mermaid(&self, player: Option<u8>) -> (String, usize) {
        let mut s = String::new();
        let mut n_links = 0;
        let mut hidden_games: HashSet<usize> = HashSet::new();
        let mut hidden_decisions: HashSet<(usize, usize)> = HashSet::new();

        fn hide_game<const MULTIPLAYER: bool, const PLAYERS: usize>(
            tree: &Tree<MULTIPLAYER, PLAYERS>,
            hidden_games: &mut HashSet<usize>,
            hidden_decisions: &mut HashSet<(usize, usize)>,
            node_i: usize,
        ) {
            hidden_games.insert(node_i);
            for (decision_i, _) in tree.nodes[node_i]
                .decisions
                .as_ref()
                .map_or(vec![], |e| e.decisions.iter().enumerate().collect())
            {
                hidden_decisions.insert((node_i, decision_i));
                hide_decision(tree, hidden_games, hidden_decisions, node_i, decision_i);
            }
        }

        fn hide_decision<const MULTIPLAYER: bool, const PLAYERS: usize>(
            tree: &Tree<MULTIPLAYER, PLAYERS>,
            hidden_games: &mut HashSet<usize>,
            hidden_decisions: &mut HashSet<(usize, usize)>,
            node_i: usize,
            decision_i: usize,
        ) {
            hidden_decisions.insert((node_i, decision_i));
            for permutation in tree.nodes[node_i].decisions.as_ref().unwrap().decisions[decision_i]
                .permutations
                .iter()
            {
                for outcome in permutation.outcomes.iter() {
                    hide_game(tree, hidden_games, hidden_decisions, outcome.node);
                }
            }
        }

        fn visit_game<const MULTIPLAYER: bool, const PLAYERS: usize>(
            tree: &Tree<MULTIPLAYER, PLAYERS>,
            player: u8,
            hidden_games: &mut HashSet<usize>,
            hidden_decisions: &mut HashSet<(usize, usize)>,
            node_i: usize,
        ) {
            let node = &tree.nodes[node_i];
            if node.decisions.is_some() {
                for (decision_i, decision) in node
                    .decisions
                    .as_ref()
                    .map_or(vec![], |e| e.decisions.iter().enumerate().collect())
                {
                    if (true || node.game().turn == player)
                        && Some(decision_i) != node.decisions.as_ref().unwrap().best
                    {
                        hide_decision(tree, hidden_games, hidden_decisions, node_i, decision_i);
                        continue;
                    }
                    for permutation in decision.permutations.iter() {
                        for outcome in permutation.outcomes.iter() {
                            visit_game(tree, player, hidden_games, hidden_decisions, outcome.node);
                        }
                    }
                }
            }
        }

        if let Some(player) = player {
            visit_game(
                &self,
                player,
                &mut hidden_games,
                &mut hidden_decisions,
                self.root,
            );
        }

        writeln!(
            s,
            "{}",
            r##"---
config:
  themeVariables:
    fontFamily: Monospace
    fontWeight: bold
    edgeLabelBackground: "#0e0e0e"
    # Node border color
    nodeBorder: "#2e2e2e"
  nodeSpacing: 25
  rankSpacing: 25
  flowchart:
    padding: 5
---"##
        )
        .unwrap();
        writeln!(s, "graph TD;").unwrap();
        for node_i in 0..self.nodes.len() {
            if hidden_games.contains(&node_i) {
                continue;
            }
            let node = &self.nodes[node_i];
            let mut game_text = node.game().clone().with_mask().to_string();

            if node
                .decisions
                .as_ref()
                .map_or(true, |e| e.decisions.is_empty())
                && !node.game().game_over()
            {
                let weights = Node::heuristic_weights(node.game());
                game_text += &weights
                    .iter()
                    .map(|w| fmt_weight(*w))
                    .collect::<Vec<_>>()
                    .join(",");
                // game_text += "\n";
                // let points = Node::all_player_points(node.game()).0;
                // game_text += &points.iter().map(|w| format!("{}", w)).collect::<Vec<_>>().join(",");
            }

            let is_me = Some(node.game().turn) == player;
            let game_color = if is_me || player.is_none() {
                "green"
            } else {
                "red"
            };
            writeln!(
                s,
                "  n{}[{:?}]",
                node_i,
                game_text.replace(" ", "&nbsp;").replace("\n", "<br>")
            )
            .unwrap();
            writeln!(
                s,
                "  style n{} text-align:left,stroke:{};",
                node_i, game_color
            )
            .unwrap();
            if let Some(player) = player {
                if node.game().game_over() {
                    if node.game().turn == player {
                        writeln!(s, "  style n{} fill:#1f9020", node_i).unwrap();
                    } else {
                        writeln!(s, "  style n{} fill:#9f2020", node_i).unwrap();
                    }
                }
            }
            // let game_weights = Node::heuristic_weights(node.game());
            let mut hidden = vec![];
            if let Some(decisions) = node.decisions.as_ref() {
                for (decision_i, decision) in decisions.decisions.iter().enumerate() {
                    let actions_strs = decision
                        .actions
                        .iter()
                        .map(|a| a.to_string(Some(node.game().turn), Some(PLAYERS)))
                        .collect::<Vec<_>>();
                    let weight = decision.weights[node.game().turn as usize];
                    if hidden_decisions.contains(&(node_i, decision_i)) {
                        hidden.push((weight, actions_strs));
                        continue;
                    }
                    let ds = format!("d{}-{}", node_i, decision_i);
                    writeln!(
                        s,
                        "  {}({:?})",
                        ds,
                        format!("{} {}", fmt_weight(weight), actions_strs.join(" "))
                    )
                    .unwrap();
                    writeln!(s, "  n{} --> {}", node_i, ds).unwrap();
                    let is_best = Some(decision_i) == decisions.best;
                    if is_best {
                        writeln!(
                            s,
                            "  style d{}-{} stroke: {}",
                            node_i, decision_i, game_color
                        )
                        .unwrap();
                        writeln!(s, "  linkStyle {} stroke: {};", n_links, game_color).unwrap();
                    }
                    n_links += 1;
                    for (k, permutation) in decision.permutations.iter().enumerate() {
                        fn write_outcomes<const PLAYERS: usize>(
                            s: &mut String,
                            n_links: &mut usize,
                            parent: &str,
                            outcomes: &SmallVec<[Outcome<PLAYERS>; 2]>,
                        ) {
                            for outcome in outcomes.iter() {
                                let p = if outcome.chance.iter().all(|c| *c == outcome.chance[0]) {
                                    if (outcome.chance[0] - 1.0).abs() < EPSILON {
                                        None
                                    } else {
                                        Some(fmt_percent(outcome.chance[0]))
                                    }
                                } else {
                                    outcome
                                        .chance
                                        .iter()
                                        .map(|c| fmt_percent(*c))
                                        .collect::<Vec<_>>()
                                        .join(",")
                                        .into()
                                };
                                if let Some(p) = p {
                                    writeln!(s, "  {} -->|{:?}| n{}", parent, p, outcome.node)
                                        .unwrap();
                                } else {
                                    writeln!(s, "  {} --> n{}", parent, outcome.node).unwrap();
                                }
                                *n_links += 1;
                            }
                        }

                        if let Some((shell, is_live, swap)) = permutation.reveal {
                            let ps = format!(
                                "{} {}",
                                if shell == 0 {
                                    "".to_string()
                                } else {
                                    format!("{} ", shell)
                                },
                                if is_live { "live" } else { "blank" }
                            );
                            let pn = format!("p{}-{}-{}", node_i, decision_i, k);
                            writeln!(s, "  {}({:?})", pn, ps).unwrap();
                            writeln!(s, "  {} --> {}", ds, pn).unwrap();
                            n_links += 1;
                            write_outcomes(&mut s, &mut n_links, &pn, &permutation.outcomes);
                        } else {
                            write_outcomes(&mut s, &mut n_links, &ds, &permutation.outcomes);
                        }

                        // let new_weights = Node::heuristic_weights(&self.nodes[outcome.node].key.game);
                    }
                }
            }
            if hidden.len() > 0 {
                hidden.sort_by_key(|(w, _)| (*w * -1000000.0) as i32);
                writeln!(
                    s,
                    "  h{}({:?})",
                    node_i,
                    hidden
                        .iter()
                        .map(|(w, a)| format!("{} {}", fmt_weight(*w), a.join(" ")))
                        .collect::<Vec<_>>()
                        .join("<br>")
                )
                .unwrap();
                writeln!(s, "  n{} --> h{}", node_i, node_i).unwrap();
                n_links += 1;
                writeln!(s, "  style h{} text-align:left", node_i).unwrap();
            }
        }
        (s, n_links)
    }
}
