use crate::state::{
    fmt_percent, Action, BitArray, Game, GameConfig, Item, ItemAction, ItemMap, Player,
};
use slab::Slab;
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;
use std::ops::Mul;

mod state;

fn main() {
    let game: Game<false, 2> = Game {
        max_health: 4,
        players: [
            Player {
                health: 1,
                items: ItemMap::new().with(Item::ExpiredMedicine, 1),
                known_shells: BitArray(0b01),
            },
            Player {
                health: 1,
                items: ItemMap::new(),
                known_shells: BitArray(0b00),
            },
        ],
        handcuffed: BitArray(0b00),
        live_shells: BitArray(0b00),
        n_shells: 2,
        turn: 0,
        sawed_off: false,
        forward: true,
    };

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

struct Decision<const PLAYERS: usize> {
    preferences: [f32; PLAYERS],
    depth: usize,
    actions: SmallVec<[Action; 6]>,
    outcomes: SmallVec<[(f32, usize); 3]>,
}

impl<const PLAYERS: usize> Decision<PLAYERS> {
    pub fn new_empty() -> Self {
        Decision {
            preferences: [0.0; PLAYERS],
            depth: 0,
            actions: smallvec![],
            outcomes: smallvec![],
        }
    }

    pub fn new(actions: SmallVec<[Action; 6]>) -> Self {
        Decision {
            preferences: [0.0; PLAYERS],
            depth: 0,
            actions,
            outcomes: smallvec![],
        }
    }

    pub fn new_single(actions: SmallVec<[Action; 6]>, outcome: usize) -> Self {
        Decision {
            preferences: [0.0; PLAYERS],
            depth: 0,
            actions,
            outcomes: smallvec![(1.0, outcome)],
        }
    }

    pub fn new_multiple(
        actions: SmallVec<[Action; 6]>,
        outcomes: SmallVec<[(f32, usize); 3]>,
    ) -> Self {
        Decision {
            preferences: [0.0; PLAYERS],
            depth: 0,
            actions,
            outcomes,
        }
    }
}

struct Node<const MULTIPLAYER: bool, const PLAYERS: usize> {
    pub game: Game<MULTIPLAYER, PLAYERS>,
    pub preferences: [f32; PLAYERS],
    pub decisions: Option<SmallVec<[Decision<PLAYERS>; 3]>>,
}

impl<const MULTIPLAYER: bool, const PLAYERS: usize> Node<MULTIPLAYER, PLAYERS> {
    pub fn new(game: Game<MULTIPLAYER, PLAYERS>) -> Self {
        let preferences = Self::all_player_preferences(&game);
        Node {
            game,
            preferences,
            decisions: None,
        }
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

    fn all_player_preferences(game: &Game<MULTIPLAYER, PLAYERS>) -> [f32; PLAYERS] {
        let (points, total) = Self::all_player_points(game);
        let mut preferences = [0.0; PLAYERS];
        for i in 0..PLAYERS {
            preferences[i] = (points[i] * (PLAYERS as i32 - 1) - (total - points[i])) as f32;
        }
        preferences
    }
}

struct Tree<const MULTIPLAYER: bool, const PLAYERS: usize> {
    nodes: Slab<Node<MULTIPLAYER, PLAYERS>>,
    root: usize,
}
impl<const MULTIPLAYER: bool, const PLAYERS: usize> Tree<MULTIPLAYER, PLAYERS> {
    pub fn new(base_game: &Game<MULTIPLAYER, PLAYERS>) -> Self {
        let mut nodes = Slab::new();
        let root = nodes.insert(Node::new(base_game.clone()));
        Tree {
            nodes,
            root,
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

    pub fn add_game(&mut self, game: Game<MULTIPLAYER, PLAYERS>) -> usize {
        let mut node = Node::new(game);
        if node.game.game_over() {
            node.decisions = Some(SmallVec::new());
        }
        self.nodes.insert(node)
    }

    pub fn compute_actions(&mut self, node: usize) {
        let game = self.nodes[node].game.clone();
        let mut decisions = SmallVec::new();
        game.visit_certain(&mut |actions, game, done| {
            if done {
                decisions.push(Decision::new_single(
                    actions.into(),
                    self.add_game(game.clone()),
                ));
            } else {
                let mut outcomes = smallvec![];
                game.clone().apply_action(
                    actions.last().unwrap().clone(),
                    1.0,
                    &mut |game, chance| {
                        outcomes.push((chance, self.add_game(game)));
                    },
                );
                decisions.push(Decision::new_multiple(actions.into(), outcomes))
            }
        });
        self.nodes[node].decisions = Some(decisions);
    }

    pub fn get_node_preferences(
        &mut self,
        node: usize,
        max_depth: usize,
    ) -> [f32; PLAYERS] {
        let turn = self.nodes[node].game.turn;
        if max_depth == 0 {
            return self.nodes[node].preferences;
        }
        if self.nodes[node].decisions.is_none() {
            self.compute_actions(node);
        }
        let decisions = self.nodes[node].decisions.as_ref().unwrap();
        let n_decisions = decisions.len();
        if decisions.len() == 0 {
            return self.nodes[node].preferences;
        }
        let mut best_decision = 0;
        let mut best_preferences = decisions[0].preferences;
        for i in 1..n_decisions {
            let preferences = self.get_decision_preferences(node, i, max_depth - 1);
            if preferences[turn as usize] > best_preferences[turn as usize] {
                best_decision = i;
                best_preferences = preferences;
            }
        }
        best_preferences
    }

    pub fn get_decision_preferences(
        &mut self,
        node: usize,
        decision: usize,
        max_depth: usize,
    ) -> [f32; PLAYERS] {
        let d2 = &mut self.nodes[node].decisions.as_mut().unwrap()[decision];
        if d2.depth >= max_depth {
            return d2.preferences;
        }
        let mut d = Decision::new_empty();
        std::mem::swap(&mut d, d2);
        let mut preferences = [0.0; PLAYERS];
        for (chance, outcome) in &d.outcomes {
            let node_preferences =
                self.get_node_preferences(*outcome, max_depth);
            for i in 0..PLAYERS {
                preferences[i] += node_preferences[i] * chance;
            }
        }
        d.preferences = preferences;
        std::mem::swap(&mut d, &mut self.nodes[node].decisions.as_mut().unwrap()[decision]);
        preferences
    }
}
