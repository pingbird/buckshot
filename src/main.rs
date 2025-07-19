#![allow(dead_code)]
#![allow(unused_imports)]

use crate::state::{
    fmt_percent, Action, BitArray, Game, GameConfig, Item, ItemAction, ItemMap, Player,
};
use slab::Slab;
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;
use std::fmt::Write;
use std::ops::Mul;

mod state;

fn main() {
    let game: Game<false, 2> = Game {
        max_health: 4,
        players: [
            Player {
                health: 1,
                items: ItemMap::new(),
                known_shells: BitArray(0b00),
            },
            Player {
                health: 1,
                items: ItemMap::new(),
                known_shells: BitArray(0b00),
            },
        ],
        handcuffed: BitArray(0b00),
        live_shells: BitArray(0b01),
        n_shells: 2,
        turn: 0,
        sawed_off: false,
        forward: true,
    };

    let mut tree = Tree::new(&game);
    tree.get_decisions(tree.root, 2);
    // Run mmdc -i output.mmd -o output.png and then open output.png in chrome
    let s = tree.to_mermaid(Some(0));
    std::fs::write("output.mmd", s).unwrap();
    std::process::Command::new(r"C:\Users\ping\AppData\Roaming\npm\mmdc.cmd")
        .args([
            "-i",
            "output.mmd",
            "-o",
            "output.png",
            "-t",
            "dark",
            "-b",
            "#0e0e0e",
            "-s",
            "10",
        ])
        .output()
        .unwrap();
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

// 8 bits per shell, 0-255 -> 0-1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct KnownChance(u64);

impl KnownChance {
    pub fn new() -> Self {
        KnownChance(0)
    }

    pub fn pop_front(&mut self) {
        self.0 = self.0 >> 8;
    }

    pub fn get_raw(&self, index: u8) -> u8 {
        ((self.0 >> (index * 8)) & 0b11111111) as u8
    }

    pub fn set_raw(&mut self, index: u8, value: u8) {
        self.0 =
            (self.0 & !(0b11111111 << (index * 8))) | ((value as u64) << (index * 8));
    }

    pub fn get(&self, index: u8) -> f32 {
        self.get_raw(index) as f32 / 255.0
    }

    pub fn set(&mut self, index: u8, value: f32) {
        self.set_raw(index, (value * 255.0) as u8);
    }

    pub fn add_chance(&mut self, index: u8, chance: f32) {
        if chance == 0.0 {
            return;
        } else if chance == 1.0 {
            self.set(index, 1.0);
        } else {
            let current = self.get(index);
            self.set(index, current + chance * (1.0 - current));
        }
    }
}

struct Outcome<const PLAYERS: usize> {
    chance: [f32; PLAYERS],
    node: usize,
}

struct Decision<const PLAYERS: usize> {
    action: Action,
    weights: [f32; PLAYERS],
    outcomes: SmallVec<[Outcome<PLAYERS>; 4]>,
}

struct Decisions<const PLAYERS: usize> {
    decisions: SmallVec<[Decision<PLAYERS>; 4]>,
    depth: usize,
    best: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key<const MULTIPLAYER: bool, const PLAYERS: usize> {
    pub game: Game<MULTIPLAYER, PLAYERS>,
    pub known_shells: BitArray<u8>,
    pub known_chance: [KnownChance; PLAYERS],
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

    fn known_shells(&self) -> BitArray<u8> {
        self.key.known_shells
    }

    fn known_chance(&self) -> &[KnownChance; PLAYERS] {
        &self.key.known_chance
    }

    fn player_points(game: &Game<MULTIPLAYER, PLAYERS>, player_index: u8) -> i32 {
        let mut points: i32 = 0;
        let player = &game.players[player_index as usize];
        points += (player.health as i32) * 8;
        for item in Item::ALL {
            points += *player.items.get(item) as i32
        }
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
        let known_shells = base_game.any_known_shells();
        let mut nodes = Slab::new();
        let key = Key {
            game: base_game,
            known_shells: known_shells,
            known_chance: [KnownChance::new(); PLAYERS],
        };
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
        // if let Some(node) = self.nodes_by_key.get(&key) {
        //     *node
        // } else {
            let node = Node {
                key: key.clone(),
                decisions: None,
            };
            let node_index = self.nodes.insert(node);
            self.nodes_by_key.insert(key, node_index);
            node_index
        // }
    }

    fn apply_action(
        &mut self,
        key: Key<MULTIPLAYER, PLAYERS>,
        action: &Action,
        outcomes: &mut SmallVec<[Outcome<PLAYERS>; 4]>,
        weights: &mut [f32; PLAYERS],
        max_depth: usize,
        chance: [f32; PLAYERS],
    ) {
        key.game.apply_action(action.clone(), |game, outcome_chance| {
            let key = Key {
                game: game.clone(),
                known_shells: key.known_shells,
                known_chance: key.known_chance,
            };
            let outcome = self.add_game(key);
            outcomes.push(Outcome {
                chance: chance.map(|c| c * outcome_chance),
                node: outcome,
            });
            if max_depth == 0 {
                // Use heuristics
                let heuristic_weights = Node::heuristic_weights(&self.nodes[outcome].key.game);
                for i in 0..PLAYERS {
                    weights[i] += heuristic_weights[i] * outcome_chance;
                }
                return;
            }
            let outcome_decisions = self.get_decisions(outcome, max_depth - 1);
            match outcome_decisions.best {
                None => {
                    // Game over, winner takes all
                    let winner = self.nodes[outcome].key.game.turn;
                    weights[winner as usize] += outcome_chance;
                }
                Some(best) => {
                    // Assume the best decision will be taken
                    let best_decision = &outcome_decisions.decisions[best];
                    for j in 0..PLAYERS {
                        weights[j] += best_decision.weights[j] * outcome_chance;
                    }
                }
            }
        });
    }

    pub fn compute_decisions(&mut self, node: usize, max_depth: usize) -> Decisions<PLAYERS> {
        eprintln!("compute_decisions: {}", node);
        let game = self.nodes[node].game().clone();
        let known_shells = self.nodes[node].known_shells();
        let known_chance = self.nodes[node].known_chance().clone();
        let turn = game.turn;
        let mut decisions = SmallVec::new();
        let mut best_weight = f32::MIN;
        let mut best = None;
        game.visit_actions(&mut |game: &Game<MULTIPLAYER, PLAYERS>, action: Action| {
            let mut outcomes = smallvec![];
            let mut weights = [0.0; PLAYERS];
            
            let uses_shell = game.action_uses_shell(&action);
            let mut known_shells = known_shells.clone();
            let mut known_chance = known_chance.clone();
            if uses_shell {
                known_shells.pop_front();
                for i in 0..PLAYERS {
                    known_chance[i].pop_front();
                }
            }

            let mut reveal_permutations: Vec<(u8, f32)> = vec![];
            game.visit_action_reveal_chance(&action, |_, player, shell, chance| {
                reveal_permutations.push((shell, chance));
                if let Some(player) = player {
                    known_chance[player as usize].add_chance(shell, chance);
                }
            });
            eprintln!("reveal_permutations: {:?}", reveal_permutations);

            if reveal_permutations.is_empty() {
                // Just apply the action
                self.apply_action(
                    Key {
                        game: game.clone(),
                        known_shells: known_shells,
                        known_chance: known_chance,
                    },
                    &action,
                    &mut outcomes,
                    &mut weights,
                    max_depth,
                    [1.0; PLAYERS],
                );
            } else {
                for (shell, chance) in reveal_permutations {
                    let mut known_shells = known_shells.clone();
                    known_shells.set(shell, true);
                    let mut correct_chance = [0.0; PLAYERS];
                    for i in 0..PLAYERS {
                        correct_chance[i] = game.live_chance(shell, game.players[i].known_shells) * chance;
                    }
                    let (correct_chance, incorrect_chance) = if game.live_shells.get(shell) {
                        (correct_chance, correct_chance.map(|c| 1.0 - c))
                    } else {
                        (correct_chance.map(|c| 1.0 - c), correct_chance)
                    };
                    eprintln!("correct_chance: {:?}", correct_chance);
                    self.apply_action(
                        Key {
                            game: game.clone(),
                            known_shells: known_shells,
                            known_chance: known_chance,
                        },
                        &action,
                        &mut outcomes,
                        &mut weights,
                        max_depth,
                        correct_chance,
                    );
                    if game.can_flip(shell, known_shells) {
                        self.apply_action(
                            Key {
                                game: game.clone().flip(shell, known_shells).clone(),
                                known_shells: known_shells,
                                known_chance: known_chance,
                            },
                            &action,
                            &mut outcomes,
                            &mut weights,
                            max_depth,
                            incorrect_chance,
                        );
                    }
                }
            }

            let weight = weights[turn as usize];
            if weight > best_weight {
                best_weight = weight;
                best = Some(decisions.len());
            }
            decisions.push(Decision {
                action,
                weights,
                outcomes,
            })
        });
        Decisions {
            decisions: decisions,
            depth: max_depth,
            best: best,
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

    pub fn to_mermaid(&self, player: Option<u8>) -> String {
        let mut s = String::new();
        writeln!(s, "{}", r##"---
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
---"##).unwrap();
        writeln!(s, "graph TD;").unwrap();
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            writeln!(
                s,
                "  n{}[{:?}]",
                i,
                node.game()
                    .clone()
                    .with_known(node.known_shells())
                    .to_string()
                    .replace(" ", "&nbsp;")
                    .replace("\n", "<br>")
            )
            .unwrap();
            writeln!(s, "  style n{} text-align:left", i).unwrap();
            if let Some(player) = player {
                if node.game().game_over() {
                    if node.game().turn == player {
                        writeln!(s, "  style n{} fill:#1f9020", i).unwrap();
                    } else {
                        writeln!(s, "  style n{} fill:#9f2020", i).unwrap();
                    }
                }
            }
            // let game_weights = Node::heuristic_weights(node.game());
            if let Some(decisions) = node.decisions.as_ref() {
                for (j, decision) in decisions.decisions.iter().enumerate() {
                    writeln!(s, "  d{}-{}({:?})", i, j, decision.action.to_string()).unwrap();
                    writeln!(s, "  style d{}-{} border-radius:10px", i, j).unwrap();
                    writeln!(s, "  n{} --> d{}-{}", i, i, j).unwrap();
                    for outcome in decision.outcomes.iter() {
                        // let new_weights = Node::heuristic_weights(&self.nodes[outcome.node].key.game);
                        let p = if outcome.chance.iter().all(|c| *c == outcome.chance[0]) {
                            fmt_percent(outcome.chance[0])
                        } else {
                            outcome.chance.iter().map(|c| fmt_percent(*c)).collect::<Vec<_>>().join(",")
                        };
                        writeln!(s, "  d{}-{} -->|{:?}| n{}", i, j, p, outcome.node).unwrap();
                    }
                }
            }
        }
        s
    }
}
