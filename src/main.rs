use crate::state::{
    fmt_percent, Action, BitArray, Game, GameConfig, Item, ItemAction, ItemMap, Player,
};
use slab::Slab;
use std::collections::HashMap;

mod state;

fn main() {
    let game: Game<false, 2> = Game {
        max_health: 2,
        players: [
            Player {
                health: 2,
                items: ItemMap::new()
                    .with(Item::Inverter, 1)
                    .with(Item::Cigarette, 1)
                    .with(Item::BurnerPhone, 1)
                    .with(Item::ExpiredMedicine, 1),
                known_shells: BitArray(0b0000),
            },
            Player {
                health: 2,
                items: ItemMap::new()
                    .with(Item::MagnifyingGlass, 1)
                    .with(Item::Cigarette, 1)
                    .with(Item::ExpiredMedicine, 1)
                    .with(Item::Beer, 1),
                known_shells: BitArray(0b0000),
            },
        ],
        handcuffed: BitArray(0b00),
        live_shells: BitArray(0b0011),
        n_shells: 4,
        turn: 0,
        sawed_off: false,
        forward: true,
    };

    println!("{}", game);

    let mut actions = vec![];
    game.visit_certain(&mut actions, &mut |actions, game| {
        // println!("{}", game);
        println!(
            "{}",
            actions
                .iter()
                .map(|e| format!("{}", e))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("{}", "=".repeat(10));
    });
}

struct Node<const MULTIPLAYER: bool, const PLAYERS: usize> {
    pub refs: usize,
    pub game: Game<MULTIPLAYER, PLAYERS>,
    pub actions: Option<Vec<(Action, Option<Vec<(f32, usize)>>)>>,
}

struct Tree<const MULTIPLAYER: bool, const PLAYERS: usize> {
    nodes: Slab<Node<MULTIPLAYER, PLAYERS>>,
    roots: Vec<usize>,
    map: HashMap<Game<MULTIPLAYER, PLAYERS>, usize>,
}
