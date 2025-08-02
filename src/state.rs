use bitintr::{Pdep, Pext};
use lazy_static::lazy_static;
use num_traits::{PrimInt, WrappingSub};
use slab::Iter;
use smallvec::{smallvec, SmallVec};
use std::array;
use std::fmt::{format, Display, Formatter};
use std::ops::{BitAnd, BitOr, Not, Range};
use std::thread::sleep;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Item {
    ExpiredMedicine = 0,
    Inverter,
    Cigarette,
    BurnerPhone,
    MagnifyingGlass,
    Beer,
    Remote,
    HandSaw,
    Handcuffs,
    Adrenaline,
}

use Item::{
    Adrenaline, Beer, BurnerPhone, Cigarette, ExpiredMedicine, HandSaw, Handcuffs, Inverter,
    MagnifyingGlass, Remote,
};

impl Item {
    pub const COUNT: usize = 10;

    pub const ALL: [Item; Self::COUNT] = [
        ExpiredMedicine,
        Inverter,
        Cigarette,
        BurnerPhone,
        MagnifyingGlass,
        Beer,
        Remote,
        HandSaw,
        Handcuffs,
        Adrenaline,
    ];

    pub const AFFECTED_BY: [&'static [Item]; Self::COUNT] = [
        // ExpiredMedicine
        &[],
        // Inverter
        &[],
        // Cigarette
        &[],
        // BurnerPhone
        &[],
        // MagnifyingGlass
        &[],
        // Beer
        &[],
        // Remote
        &[],
        // HandSaw
        &[],
        // Handcuffs
        &[],
        // Adrenaline
        &[],
    ];
}

impl Display for Item {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let short = match self {
            ExpiredMedicine => "Med",
            Inverter => "Inv",
            Cigarette => "Cig",
            BurnerPhone => "Phn",
            MagnifyingGlass => "Mag",
            Beer => "Brr",
            HandSaw => "Saw",
            Remote => "Rem",
            Handcuffs => "Cuf",
            Adrenaline => "Adr",
        };
        write!(f, "{}", short)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemSet(pub u16);

impl ItemSet {
    pub const ALL: ItemSet = ItemSet((1 << Item::COUNT) - 1);

    pub fn new() -> Self {
        ItemSet(0)
    }

    pub fn from_item(item: Item) -> Self {
        let mut set = ItemSet::new();
        set.add(item);
        set
    }

    pub fn add(&mut self, item: Item) {
        self.0 |= 1 << (item as u16);
    }

    pub fn remove(&mut self, item: Item) {
        self.0 &= !(1 << (item as u16));
    }

    pub fn contains(&self, item: Item) -> bool {
        (self.0 & (1 << (item as u16))) != 0
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub fn union(&self, other: &ItemSet) -> ItemSet {
        ItemSet(self.0 | other.0)
    }

    pub fn intersection(&self, other: ItemSet) -> ItemSet {
        ItemSet(self.0 & other.0)
    }

    pub fn clear(&mut self) {
        self.0 = 0;
    }

    pub fn count(&self) -> u32 {
        self.0.count_ones()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ItemMap<V>(pub [V; Item::COUNT]);

impl<V> ItemMap<V> {
    pub fn new() -> Self
    where
        V: Default,
    {
        ItemMap(Default::default())
    }

    pub fn get(&self, item: Item) -> &V {
        &self.0[item as usize]
    }

    pub fn set(&mut self, item: Item, value: V) {
        self.0[item as usize] = value;
    }

    pub fn with(mut self, item: Item, value: V) -> ItemMap<V> {
        self.set(item, value);
        self
    }
}

impl Display for ItemMap<u8> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for item in Item::ALL {
            for _ in 0..*self.get(item) {
                if !first {
                    write!(f, " ")?;
                }
                write!(f, "{}", item)?;
                first = false;
            }
        }
        Ok(())
    }
}

pub trait ItemCountMap {
    fn to_set(&self) -> ItemSet;
    fn sum(&self) -> u8;
    fn contains(&self, item: Item) -> bool;
    fn increment(&mut self, item: Item);
    fn decrement(&mut self, item: Item);
    fn clear(&mut self);
}

impl ItemCountMap for ItemMap<u8> {
    fn to_set(&self) -> ItemSet {
        let mut set = ItemSet::new();
        for item in Item::ALL {
            if self.get(item) > &0 {
                set.add(item);
            }
        }
        set
    }

    fn sum(&self) -> u8 {
        let mut total = 0;
        for item in Item::ALL {
            total += self.get(item);
        }
        total
    }

    fn contains(&self, item: Item) -> bool {
        self.get(item) > &0
    }

    fn increment(&mut self, item: Item) {
        self.0[item as usize] += 1;
    }

    fn decrement(&mut self, item: Item) {
        self.0[item as usize] -= 1;
    }

    fn clear(&mut self) {
        for item in Item::ALL {
            self.set(item, 0);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitArray<T: PrimInt + WrappingSub>(pub T);

impl<T: PrimInt + WrappingSub + TryFrom<u64>> BitArray<T> {
    pub fn zero() -> Self {
        BitArray(T::zero())
    }

    pub fn all(len: u8) -> Self {
        BitArray(
            T::from(
                T::one()
                    .to_u64()
                    .unwrap()
                    .unbounded_shl(len as u32)
                    .wrapping_sub(1),
            )
            .unwrap(),
        )
    }

    pub fn single(index: u8) -> Self {
        BitArray(T::one() << index as usize)
    }

    pub fn get(&self, index: u8) -> bool {
        (self.0 >> index as usize) & T::one() != T::zero()
    }

    pub fn set(&mut self, index: u8, value: bool) {
        self.0 = (self.0 & !(T::one() << index as usize))
            | if value {
                T::one() << index as usize
            } else {
                T::zero()
            };
    }

    pub fn flip(&mut self, index: u8) {
        self.0 = self.0 ^ T::one() << index as usize;
    }

    pub fn front(&self) -> bool {
        self.get(0)
    }

    pub fn pop_front(&mut self) {
        self.0 = self.0 >> 1;
    }

    pub fn count(&self) -> u32 {
        self.0.count_ones()
    }

    pub fn count_unset(&self, len: u32) -> u32 {
        len - self.0.count_ones()
    }

    pub fn empty(&self) -> bool {
        self.0 == T::zero()
    }

    pub fn union(&self, other: BitArray<T>) -> BitArray<T> {
        BitArray(self.0 | other.0)
    }

    pub fn intersection(&self, other: BitArray<T>) -> BitArray<T> {
        BitArray(self.0 & other.0)
    }
}

impl BitOr for BitArray<u8> {
    type Output = Self;

    fn bitor(self, other: Self) -> Self::Output {
        BitArray(self.0 | other.0)
    }
}

impl BitAnd for BitArray<u8> {
    type Output = Self;
    fn bitand(self, other: Self) -> Self::Output {
        BitArray(self.0 & other.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Player {
    pub health: u8,
    pub items: ItemMap<u8>,
    pub known_shells: BitArray<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GameConfig {
    pub multiplayer: bool,
    pub n_players: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Game<const MULTIPLAYER: bool, const PLAYERS: usize> {
    pub max_health: u8,
    pub players: [Player; PLAYERS],
    pub handcuffed: BitArray<u8>,
    pub live_shells: BitArray<u8>,
    pub n_shells: u8,
    pub turn: u8,
    pub sawed_off: bool,
    pub forward: bool,
}

// 8 bits per shell, 0-255 -> 0-1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KnownChance(u64);

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
        self.0 = (self.0 & !(0b11111111 << (index * 8))) | ((value as u64) << (index * 8));
    }

    pub fn get(&self, index: u8) -> f32 {
        self.get_raw(index) as f32 / 255.0
    }

    pub fn set(&mut self, index: u8, value: f32) {
        self.set_raw(index, (value * 255.0) as u8);
    }

    pub fn with(self, index: u8, value: f32) -> Self {
        let mut new = self.clone();
        new.set(index, value);
        new
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

pub struct GameWithMask<const MULTIPLAYER: bool, const PLAYERS: usize> {
    pub game: Game<MULTIPLAYER, PLAYERS>,
    pub is_masked: bool,
}

impl<const MULTIPLAYER: bool, const PLAYERS: usize> Display for GameWithMask<MULTIPLAYER, PLAYERS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let game = &self.game;
        write!(f, "|")?;
        for i in 0..game.n_shells {
            if !self.is_masked || game.any_known_shells().get(i) {
                if game.live_shells.get(i) {
                    write!(f, "#")?;
                } else {
                    write!(f, ".")?;
                }
            } else if self.is_masked && game.live_shells.get(i) {
                write!(f, "?")?;
            } else {
                write!(f, " ")?;
            }
        }
        write!(f, "| ")?;
        if MULTIPLAYER {
            if game.forward {
                write!(f, "cw ")?;
            } else {
                write!(f, "ccw ")?;
            }
        }
        if game.sawed_off {
            write!(f, "x2 ")?;
        }
        writeln!(f, "")?;

        for player in 0..PLAYERS {
            if game.turn as usize == player {
                write!(f, ">")?;
            } else if game.handcuffed.get(player as u8) {
                write!(f, "&")?;
            } else if game.players[player].health == 0 {
                write!(f, "X")?;
            } else {
                write!(f, " ")?;
            }
            write!(f, " ")?;
            for h in 0..game.max_health {
                if game.players[player].health > h {
                    write!(f, "♥")?;
                } else {
                    write!(f, " ")?;
                }
            }
            if game.n_shells > 0 && game.players[player].health > 0 {
                write!(f, " ")?;
                write!(f, "|")?;
                for i in 0..game.n_shells {
                    if game.players[player].known_shells.get(i) {
                        if game.live_shells.get(i) {
                            write!(f, "#")?;
                        } else {
                            write!(f, ".")?;
                        }
                    } else {
                        write!(f, " ")?;
                    }
                }
                write!(f, "|")?;
            }
            write!(f, " ")?;
            for item in Item::ALL {
                for _ in 0..*game.players[player].items.get(item) {
                    write!(f, "{} ", item)?;
                }
            }
            writeln!(f, "")?;
        }
        Ok(())
    }
}

impl<const MULTIPLAYER: bool, const PLAYERS: usize> Display for Game<MULTIPLAYER, PLAYERS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            GameWithMask {
                game: self.clone(),
                is_masked: false,
            }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemAction {
    UseSimple(Item),
    StealSimple(u8, Item),
    StealHandcuffs(u8, u8),
    Handcuff(u8),
}

impl Display for ItemAction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string(None),)?;
        Ok(())
    }
}

impl ItemAction {
    pub fn to_string(&self, n_players: Option<usize>) -> String {
        match self {
            ItemAction::UseSimple(Item::Adrenaline) => format!("Adr _"),
            ItemAction::UseSimple(item) => format!("{}", item),
            ItemAction::StealSimple(_, item) if n_players == Some(2) => format!("Adr {}", item),
            ItemAction::StealSimple(player, item) => format!("Adr{} {}", player, item),
            ItemAction::StealHandcuffs(_, _) if n_players == Some(2) => "Adr Cuf".to_string(),
            ItemAction::StealHandcuffs(player, target) => format!("Adr{} Cuf{}", player, target),
            ItemAction::Handcuff(_) if n_players == Some(2) => "Cuf".to_string(),
            ItemAction::Handcuff(target) => format!("Cuf{}", target),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Item(ItemAction),
    Shoot(u8),
}

impl Display for Action {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string(None, None),)?;
        Ok(())
    }
}

impl Action {
    pub fn to_string(&self, player: Option<u8>, n_players: Option<usize>) -> String {
        match self {
            Action::Item(action) => action.to_string(n_players),
            Action::Shoot(target) if Some(*target) == player => "!s".to_string(),
            Action::Shoot(target) if n_players == Some(2) && Some(*target) != player => {
                "!".to_string()
            }
            Action::Shoot(target) => format!("!{}", target),
        }
    }
}

type CoverLut = [Vec<u8>; 9];

pub fn gen_lut() -> CoverLut {
    let mut lut: [Vec<u8>; 9] = Default::default();
    for i in 0..=u8::MAX {
        lut[i.count_ones() as usize].push(i);
    }
    lut
}

#[derive(Debug, Clone)]
pub struct Gosper(u8, u8);

impl Gosper {
    pub fn new(k: u8, n: u8) -> Self {
        assert!(k <= n);
        let start = 1u8.unbounded_shl(k as u32).wrapping_sub(1u8);
        Gosper(start, start.unbounded_shl((n - k) as u32))
    }
}

impl Iterator for Gosper {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.0;
        if next < self.1 {
            let c = next & 0.wrapping_sub(&next);
            let r = next.wrapping_add(c);
            self.0 = ((((r ^ next) as i8 >> 2) / (c as i8)) | (r as i8)) as u8;
            Some(next)
        } else if self.1 == 0 {
            None
        } else {
            let last = self.1;
            self.1 = 0;
            Some(last)
        }
    }
}

lazy_static! {
    static ref LUT: CoverLut = gen_lut();
}

pub fn min_cover_inner<const S: usize>(sets: [u8; S]) -> u8 {
    let lut = &LUT;
    let any = sets.iter().fold(0, u8::bitor);
    let masks = sets.map(|e| (e as u32).pext(any as u32) as u8);
    let max_k = any.count_ones();
    let max_i = 1u8.unbounded_shl(max_k).wrapping_sub(1u8);
    for k in 1..any.count_ones() {
        'gos: for i in &lut[k as usize] {
            for mask in &masks {
                if i & mask == 0 {
                    if *i >= max_i {
                        break 'gos;
                    }
                    continue 'gos;
                }
            }
            return (*i as u32).pdep(any as u32) as u8;
        }
    }
    any
}

pub struct RevealPermutation<const MULTIPLAYER: bool, const PLAYERS: usize> {
    pub reveal: Option<(u8, Option<u8>)>,
    pub outcomes: SmallVec<[(Game<MULTIPLAYER, PLAYERS>, [f32; PLAYERS]); 2]>,
}

impl<const MULTIPLAYER: bool, const PLAYERS: usize> RevealPermutation<MULTIPLAYER, PLAYERS> {
    pub fn expand<
        F: FnMut(Game<MULTIPLAYER, PLAYERS>) -> SmallVec<[(Game<MULTIPLAYER, PLAYERS>, f32); 2]>,
    >(
        self,
        mut f: F,
    ) -> Self {
        let mut outcomes = smallvec![];
        for (game, chance) in self.outcomes {
            for (outcome, outcome_chance) in f(game) {
                outcomes.push((outcome, array::from_fn(|i| chance[i] * outcome_chance)));
            }
        }
        RevealPermutation {
            reveal: self.reveal,
            outcomes,
        }
    }

    pub fn map<F: FnMut(Game<MULTIPLAYER, PLAYERS>) -> Game<MULTIPLAYER, PLAYERS>>(
        self,
        mut f: F,
    ) -> Self {
        RevealPermutation {
            reveal: self.reveal,
            outcomes: self
                .outcomes
                .into_iter()
                .map(|(game, chance)| (f(game), chance))
                .collect(),
        }
    }
}

pub fn min_cover(sets: &[u8]) -> u8 {
    if sets.is_empty() {
        return 0;
    }
    if let Ok(sets) = <[u8; 1]>::try_from(sets) {
        min_cover_inner(sets)
    } else if let Ok(sets) = <[u8; 2]>::try_from(sets) {
        min_cover_inner(sets)
    } else if let Ok(sets) = <[u8; 3]>::try_from(sets) {
        min_cover_inner(sets)
    } else if let Ok(sets) = <[u8; 4]>::try_from(sets) {
        min_cover_inner(sets)
    } else if let Ok(sets) = <[u8; 5]>::try_from(sets) {
        min_cover_inner(sets)
    } else if let Ok(sets) = <[u8; 6]>::try_from(sets) {
        min_cover_inner(sets)
    } else if let Ok(sets) = <[u8; 7]>::try_from(sets) {
        min_cover_inner(sets)
    } else if let Ok(sets) = <[u8; 8]>::try_from(sets) {
        min_cover_inner(sets)
    } else {
        panic!("Unsupported");
    }
}

impl<const MULTIPLAYER: bool, const PLAYERS: usize> Game<MULTIPLAYER, PLAYERS> {
    pub fn with_mask(self) -> GameWithMask<MULTIPLAYER, PLAYERS> {
        GameWithMask {
            game: self,
            is_masked: true,
        }
    }

    pub fn assert_valid(&self) {
        assert_eq!(
            self.live_shells.0,
            self.live_shells.0 & BitArray::<u8>::all(self.n_shells).0
        );
        for i in 0..PLAYERS {
            for j in 0..self.n_shells {
                let swaps = self.all_swaps(j, self.players[i].known_shells);
                assert_eq!(swaps, swaps & BitArray::<u8>::all(self.n_shells));
                if let Some(shell) = self.try_swap(j, self.players[i].known_shells) {
                    assert!(swaps.get(shell))
                } else {
                    assert_eq!(swaps, BitArray::zero())
                }
            }
        }
    }

    pub fn shell_mask(&self) -> u8 {
        (1u8.unbounded_shl(self.n_shells as u32)).wrapping_sub(1u8)
    }

    pub fn blank_shells(&self) -> BitArray<u8> {
        BitArray((!self.live_shells.0) & self.shell_mask())
    }

    pub fn n_live(&self) -> u32 {
        self.live_shells.count()
    }

    pub fn all_live(&self) -> bool {
        self.live_shells.count() == self.n_shells as u32
    }

    pub fn all_blank(&self) -> bool {
        self.live_shells.empty()
    }

    pub fn visit_item_actions<F: FnMut(&Self, ItemAction) -> ()>(&self, items: ItemSet, mut f: F) {
        eprintln!("items: {}", self.players[self.turn as usize].items);
        let turn_items = self.players[self.turn as usize].items.to_set();
        let allowed = turn_items.intersection(items);
        eprintln!("allowed: {:b}", allowed.0);
        eprintln!("turn_items: {:b}", turn_items.0);
        for item in [
            ExpiredMedicine,
            Inverter,
            Cigarette,
            BurnerPhone,
            MagnifyingGlass,
            Beer,
            Remote,
            Adrenaline,
        ] {
            if allowed.contains(item) {
                f(self, ItemAction::UseSimple(item))
            }
        }
        if allowed.contains(HandSaw) && !self.sawed_off {
            f(self, ItemAction::UseSimple(HandSaw));
        }
        if turn_items.contains(Adrenaline) {
            for player in 0..PLAYERS {
                if player == self.turn as usize {
                    continue;
                }
                let allowed = self.players[player].items.to_set().intersection(items);
                for item in [
                    ExpiredMedicine,
                    Inverter,
                    Cigarette,
                    BurnerPhone,
                    MagnifyingGlass,
                    Beer,
                    Remote,
                ] {
                    if allowed.contains(item) {
                        f(self, ItemAction::StealSimple(player as u8, item))
                    }
                }
                if allowed.contains(HandSaw) && !self.sawed_off {
                    f(self, ItemAction::StealSimple(player as u8, HandSaw));
                }
                if allowed.contains(Handcuffs) {
                    for target_player in 0..PLAYERS {
                        if target_player != self.turn as usize
                            && !self.handcuffed.get(target_player as u8)
                            && self.players[target_player].health > 0
                        {
                            f(
                                self,
                                ItemAction::StealHandcuffs(player as u8, target_player as u8),
                            )
                        }
                    }
                }
            }
        }
        if allowed.contains(Handcuffs) {
            for player in 0..PLAYERS {
                if player != self.turn as usize
                    && !self.handcuffed.get(player as u8)
                    && self.players[player].health > 0
                {
                    f(self, ItemAction::Handcuff(player as u8))
                }
            }
        }
    }

    pub fn visit_actions<F: FnMut(&Self, Action) -> ()>(&self, mut f: F) {
        if self.round_over() {
            return;
        }
        self.visit_item_actions(ItemSet::ALL, |game, action| f(game, Action::Item(action)));
        for i in 0..PLAYERS {
            if self.players[i].health > 0 {
                f(self, Action::Shoot(i as u8));
            }
        }
    }

    pub fn any_known_shells(&self) -> BitArray<u8> {
        let mut known_shells = self.players[0].known_shells;
        for i in 1..PLAYERS {
            known_shells.0 |= self.players[i].known_shells.0;
        }
        known_shells
    }

    pub fn all_known_shells(&self) -> BitArray<u8> {
        let mut known_shells = self.players[0].known_shells;
        for i in 1..PLAYERS {
            known_shells.0 &= self.players[i].known_shells.0;
        }
        known_shells
    }

    pub fn visit_certain_inner<F: FnMut(&[Action], &Self, Option<Action>)>(
        &self,
        actions: &mut Vec<Action>,
        f: &mut F,
    ) {
        self.assert_valid();
        eprintln!("items: {}", self.players[self.turn as usize].items);
        self.visit_actions(|game, action| {
            eprintln!("{}", action);
            actions.push(action.clone());
            let known_shells = self.all_known_shells();
            if game.is_action_certain(known_shells, action) {
                game.clone()
                    .apply_action(action.clone(), |next_game, next_chance| {
                        assert_eq!(next_chance, 1.0);
                        // Stop if the round is over or the turn has changed
                        if next_game.round_over() || next_game.turn != game.turn {
                            f(actions, &next_game, None);
                            return;
                        } else {
                            next_game.visit_certain_inner(actions, f);
                        }
                    });
            } else {
                // Stop if the action does not have a certain outcome
                f(&actions[..actions.len() - 1], game, Some(action));
            }
            actions.pop();
        });
    }

    pub fn visit_certain<F: FnMut(&[Action], &Self, Option<Action>)>(&self, f: &mut F) {
        let mut actions = Vec::new();
        self.visit_certain_inner(&mut actions, f);
    }

    pub fn heal(mut self, player: u8, amount: u8) -> Self {
        self.players[player as usize].health = self.players[player as usize]
            .health
            .saturating_add(amount)
            .min(self.max_health);
        self
    }

    pub fn damage(mut self, player: u8, amount: u8) -> Self {
        let health = self.players[player as usize].health.saturating_sub(amount);
        self.players[player as usize].health = health;
        if health == 0 {
            self.handcuffed.set(0, false);
            self.players[player as usize].items.clear();
            self.players[player as usize].known_shells = BitArray::zero();
            if player == self.turn {
                return self.end_turn();
            }
        }
        self
    }

    pub fn damage_and_end_turn(mut self, player: u8, amount: u8) -> Self {
        let turn = self.turn;
        self = self.damage(player, amount);
        if self.turn == turn {
            self.end_turn()
        } else {
            self
        }
    }

    pub fn learn_round(mut self, player: u8, shell: u8, live: bool) -> Self {
        self.players[player as usize].known_shells.set(shell, true);
        if live != self.live_shells.get(shell) {
            // If we learned a shell was wrong, swap it with the first unknown shell of the opposite kind
            let known_shells = self.players[player as usize].known_shells;
            self.swap_first(shell, known_shells)
        } else {
            self
        }
    }

    pub fn reveal_round(mut self, player: u8, shell: u8) -> Self {
        self.players[player as usize].known_shells.set(shell, true);
        self
    }

    pub fn normalize_known(mut self) -> Self {
        if self.n_shells == 0 {
            return self;
        }
        if self.n_shells == 1 {
            // Everyone should know the last shell
            let mut known_shells = BitArray::zero();
            known_shells.set(0, true);
            for player in 0..PLAYERS {
                self.players[player].known_shells = known_shells
            }
        } else if self.all_blank() || self.all_live() {
            // Everyone knows everything
            let known_shells = BitArray::all(self.n_shells);
            for player in 0..PLAYERS {
                self.players[player].known_shells = known_shells
            }
        } else {
            for player in 0..PLAYERS {
                // If the player knows all lives or all blanks, they now know all of them
                let known = self.players[player].known_shells;
                let blank_shells = self.blank_shells();
                if self.live_shells == self.live_shells & known
                    || blank_shells == blank_shells & known
                {
                    self.players[player].known_shells = BitArray::all(self.n_shells);
                }
            }
        }
        self
    }

    pub fn eject_round(mut self) -> Self {
        self.assert_valid();
        self.n_shells -= 1;
        self.live_shells.pop_front();
        self.assert_valid();
        for player in 0..PLAYERS {
            self.players[player].known_shells.pop_front();
        }
        self.sawed_off = false;
        self.normalize_known()
    }

    pub fn live_damage(&self) -> u8 {
        if self.sawed_off {
            2
        } else {
            1
        }
    }

    pub fn saw_off(mut self) -> Self {
        self.sawed_off = true;
        self
    }

    pub fn handcuff(mut self, player: u8) -> Self {
        self.handcuffed.set(player, true);
        self
    }

    pub fn remove_item(mut self, player: u8, item: Item) -> Self {
        self.players[player as usize].items.decrement(item);
        self
    }

    pub fn invert_next(mut self) -> Self {
        self.live_shells.flip(0);
        self
    }

    pub fn reverse(mut self) -> Self {
        self.forward = !self.forward;
        self
    }

    pub fn apply_simple_item<F: FnMut(Self, f32)>(self, item: Item, mut f: F) {
        let turn = self.turn;
        match item {
            ExpiredMedicine => {
                f(self.clone().damage(turn, 1), 0.5);
                f(self.heal(turn, 2), 0.5);
            }
            Inverter => {
                f(self.invert_next(), 1.0);
            }
            Cigarette => {
                f(self.heal(turn, 1), 1.0);
            }
            BurnerPhone => {
                let n_shells = self.n_shells;
                if n_shells <= 1 {
                    // Nothing happens
                    f(self, 1.0);
                } else {
                    let mut nothing_chance = 0.0;
                    let chance = 1.0 / (n_shells - 1) as f32;
                    for i in 1..n_shells {
                        if self.players[turn as usize].known_shells.get(i) {
                            nothing_chance += chance;
                        } else {
                            f(self.clone().reveal_round(turn, i), chance)
                        }
                    }
                    if nothing_chance > 0.0 {
                        f(self, nothing_chance);
                    }
                }
            }
            MagnifyingGlass => {
                f(self.reveal_round(turn, 0), 1.0);
            }
            Beer => {
                f(self.eject_round(), 1.0);
            }
            HandSaw => {
                f(self.saw_off(), 1.0);
            }
            Remote => {
                f(self.reverse(), 1.0);
            }
            Handcuffs => unreachable!(),
            Adrenaline => {
                f(self, 1.0);
            }
        }
    }

    pub fn apply_item_action<F: FnMut(Self, f32)>(self, action: ItemAction, mut f: F) {
        let turn = self.turn;
        match action {
            ItemAction::UseSimple(item) => self.remove_item(turn, item).apply_simple_item(item, f),
            ItemAction::StealSimple(player, item) => {
                self.remove_item(turn, Adrenaline)
                    .remove_item(player, item)
                    .apply_simple_item(item, f);
            }
            ItemAction::StealHandcuffs(player, handcuff_player) => {
                f(
                    self.remove_item(turn, Adrenaline)
                        .remove_item(player, Handcuffs)
                        .handcuff(handcuff_player),
                    1.0,
                );
            }
            ItemAction::Handcuff(handcuff_player) => {
                f(
                    self.remove_item(turn, Handcuffs).handcuff(handcuff_player),
                    1.0,
                );
            }
        }
    }

    pub fn apply_action<F: FnMut(Self, f32)>(self, action: Action, mut f: F) {
        match action {
            Action::Item(action) => self.apply_item_action(action, f),
            Action::Shoot(target) => {
                let is_live = self.live_shells.front();
                if is_live {
                    let damage = self.live_damage();
                    f(self.eject_round().damage_and_end_turn(target, damage), 1.0);
                } else if target == self.turn {
                    f(self.eject_round().continue_turn(), 1.0);
                } else {
                    f(self.eject_round().end_turn(), 1.0);
                }
            }
        }
    }

    fn apply_actions_inner<F: FnMut(Game<MULTIPLAYER, PLAYERS>, f32)>(
        self,
        actions: &[Action],
        chance: f32,
        f: &mut F,
    ) {
        match actions {
            [] => f(self, chance),
            [next, rest @ ..] => self.apply_action(*next, |game, chance2| {
                game.apply_actions_inner(rest, chance * chance2, f)
            }),
        }
    }

    pub fn apply_actions<F: FnMut(Game<MULTIPLAYER, PLAYERS>, f32)>(
        self,
        actions: &[Action],
        mut f: F,
    ) {
        self.apply_actions_inner(actions, 1.0, &mut f)
    }

    fn is_item_certain(&self, known_shells: BitArray<u8>, item: Item) -> bool {
        match item {
            ExpiredMedicine => false,
            Inverter => true,
            Cigarette => true,
            BurnerPhone => {
                if self.n_shells <= 1 {
                    // If there is only one shell
                    return true;
                }
                // If we know all shells but the next
                let mask = (1 << self.n_shells) - 2;
                known_shells.0 & mask == mask
            }
            MagnifyingGlass => {
                // If we know the current shell
                known_shells.front()
            }
            Beer => {
                // If we know the current shell
                known_shells.front()
            }
            Remote => true,
            HandSaw => true,
            Handcuffs => unreachable!(),
            Adrenaline => true,
        }
    }

    fn is_item_action_certain(&self, known_shells: BitArray<u8>, action: ItemAction) -> bool {
        match action {
            ItemAction::UseSimple(item) => self.is_item_certain(known_shells, item),
            ItemAction::StealSimple(_, Item::Adrenaline) => unreachable!(),
            ItemAction::StealSimple(_, item) => self.is_item_certain(known_shells, item),
            ItemAction::StealHandcuffs(_, _) => true,
            ItemAction::Handcuff(_) => true,
        }
    }

    pub fn is_action_certain(&self, known_shells: BitArray<u8>, action: Action) -> bool {
        match action {
            Action::Item(item_action) => self.is_item_action_certain(known_shells, item_action),
            Action::Shoot(_) => {
                // Shooting is certain if the player knows the current shell
                known_shells.front()
            }
        }
    }

    pub fn item_uses_shell(&self, item: Item) -> bool {
        match item {
            Inverter => true,
            BurnerPhone => true,
            MagnifyingGlass => true,
            Beer => true,
            _ => false,
        }
    }

    pub fn action_uses_shell(&self, action: Action) -> bool {
        match action {
            Action::Item(item_action) => match item_action {
                ItemAction::UseSimple(item) => self.item_uses_shell(item),
                ItemAction::StealSimple(_, item) => self.item_uses_shell(item),
                _ => false,
            },
            Action::Shoot(_) => true,
        }
    }

    pub fn visit_item_reveal_chance<F: FnMut(&Self, Option<u8>, u8, f32)>(
        &self,
        item: Item,
        mut f: F,
    ) -> bool {
        match item {
            Beer | MagnifyingGlass => {
                f(self, None, 0, 1.0);
                true
            }
            BurnerPhone => {
                let n_shells = self.n_shells;
                if n_shells > 1 {
                    let chance = 1.0 / (n_shells - 1) as f32;
                    for i in 1..n_shells {
                        f(self, Some(self.turn), i, chance)
                    }
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    pub fn visit_action_reveal_chance<F: FnMut(&Self, Option<u8>, u8, f32)>(
        &self,
        action: &Action,
        mut f: F,
    ) -> bool {
        match action {
            Action::Item(item_action) => match item_action {
                ItemAction::UseSimple(item) => self.visit_item_reveal_chance(*item, f),
                ItemAction::StealSimple(_, item) => self.visit_item_reveal_chance(*item, f),
                _ => false,
            },
            Action::Shoot(_) => {
                f(self, None, 0, 1.0);
                true
            }
        }
    }

    pub fn visit_shell_perspective<F: FnMut(RevealPermutation<MULTIPLAYER, PLAYERS>)>(
        mut self,
        shell: u8,
        chance: f32,
        mut f: F,
    ) {
        if self.all_known_shells().get(shell) {
            f(RevealPermutation {
                reveal: None,
                outcomes: smallvec![(self, [chance; PLAYERS])],
            });
        } else {
            let mut correct_chance = [0.0; PLAYERS];
            for i in 0..PLAYERS {
                correct_chance[i] = self.live_chance(shell, self.players[i].known_shells) * chance;
            }
            let is_live = self.live_shells.get(shell);
            let (correct_chance, mut incorrect_chance) = if is_live {
                (correct_chance, correct_chance.map(|c| chance - c))
            } else {
                (correct_chance.map(|c| chance - c), correct_chance)
            };
            let all_swaps = self.all_player_swaps(shell);
            let cover_flips = BitArray(min_cover(
                all_swaps
                    .iter()
                    .map(|e| e.0)
                    .filter(|e| *e != 0)
                    .collect::<Vec<_>>()
                    .as_slice(),
            ));
            f(RevealPermutation {
                reveal: Some((shell, None)),
                outcomes: smallvec![(self.clone(), correct_chance)],
            });
            self = self.flip(shell);
            for i in 0..self.n_shells {
                if !cover_flips.get(i) {
                    continue;
                }
                let mut incorrect_chance_copy = incorrect_chance.clone();
                for j in 0..PLAYERS {
                    if all_swaps[j].get(i) {
                        incorrect_chance[j] = 0.0;
                    } else {
                        incorrect_chance_copy[j] = 0.0;
                    }
                }
                f(RevealPermutation {
                    reveal: Some((shell, Some(i))),
                    outcomes: smallvec![(self.clone().flip(i), incorrect_chance_copy)],
                });
            }
        }
    }

    pub fn visit_item_perspective<F: FnMut(RevealPermutation<MULTIPLAYER, PLAYERS>)>(
        self,
        item: Item,
        f: &mut F,
    ) -> Option<Self> {
        match item {
            BurnerPhone => {
                if self.n_shells <= 1 {
                    f(RevealPermutation {
                        reveal: None,
                        outcomes: smallvec![(self, [1.0; PLAYERS])],
                    });
                    None
                } else {
                    let turn = self.turn;
                    let mut nothing_chance = 0.0;
                    let n_shells = self.n_shells;
                    let chance = 1.0 / (n_shells - 1) as f32;
                    for i in 1..n_shells {
                        if self.players[turn as usize].known_shells.get(i) {
                            nothing_chance += chance;
                        } else {
                            self.clone()
                                .visit_shell_perspective(i, chance, |shell_perm| {
                                    f(shell_perm.map(|game| {
                                        let turn = game.turn;
                                        game.reveal_round(turn, i)
                                    }))
                                });
                        }
                    }
                    if nothing_chance > 0.0 {
                        f(RevealPermutation {
                            reveal: None,
                            outcomes: smallvec![(self, [nothing_chance; PLAYERS])],
                        });
                    }
                    None
                }
            }
            MagnifyingGlass => {
                self.visit_shell_perspective(0, 1.0, |shell_perm| {
                    f(shell_perm.map(|game| {
                        let turn = game.turn;
                        game.reveal_round(turn, 0)
                    }))
                });
                None
            }
            Beer => {
                self.visit_shell_perspective(0, 1.0, |shell_perm| {
                    f(shell_perm.map(|game| game.eject_round()))
                });
                None
            }
            _ => Some(self),
        }
    }

    pub fn visit_item_action_perspective<F: FnMut(RevealPermutation<MULTIPLAYER, PLAYERS>)>(
        self,
        item: ItemAction,
        f: &mut F,
    ) -> Option<Self> {
        match item {
            ItemAction::UseSimple(item) => {
                let turn = self.turn;
                self.remove_item(turn, item).visit_item_perspective(item, f)
            }
            ItemAction::StealSimple(target, item) => {
                let turn = self.turn;
                self.remove_item(turn, Adrenaline)
                    .remove_item(target, item)
                    .visit_item_perspective(item, f)
            }
            _ => Some(self),
        }
    }

    pub fn visit_action_perspective<F: FnMut(RevealPermutation<MULTIPLAYER, PLAYERS>)>(
        self,
        action: Action,
        mut f: F,
    ) {
        match action {
            Action::Item(item_action) => {
                match self.visit_item_action_perspective(item_action, &mut f) {
                    None => (),
                    Some(game) => game.apply_action(action, |game, outcome_chance| {
                        f(RevealPermutation {
                            reveal: None,
                            outcomes: smallvec![(game, [outcome_chance; PLAYERS])],
                        })
                    }),
                }
            }
            Action::Shoot(target) => {
                self.visit_shell_perspective(0, 1.0, |shell_perm| {
                    f(shell_perm.map(|game| {
                        let is_live = game.live_shells.front();
                        if is_live {
                            let damage = game.live_damage();
                            game.eject_round().damage_and_end_turn(target, damage)
                        } else if target == game.turn {
                            game.eject_round().continue_turn()
                        } else {
                            game.eject_round().end_turn()
                        }
                    }))
                });
            }
        }
    }

    pub fn live_chance_ratio(&self, shell: u8, known: BitArray<u8>) -> (u32, u32) {
        self.assert_valid();
        assert!(shell < self.n_shells);
        if known.get(shell) {
            return if self.live_shells.get(shell) {
                (1, 1)
            } else {
                (0, 1)
            };
        }
        let known = known.0;
        let live = self.live_shells.0;
        let unknown_live = (live & !known).count_ones();
        let unknown = self.n_shells as u32 - known.count_ones();
        assert!(
            unknown > 0,
            "live {:b} known: {:b} n_shells: {}",
            live,
            known,
            self.n_shells
        );
        (unknown_live, unknown)
    }

    pub fn live_chance(&self, shell: u8, known: BitArray<u8>) -> f32 {
        let (unknown_live, unknown) = self.live_chance_ratio(shell, known);
        unknown_live as f32 / unknown as f32
    }

    pub fn flip(mut self, shell: u8) -> Self {
        let shell_mask = 1 << shell;
        self.live_shells.0 ^= shell_mask;
        self.assert_valid();
        self
    }

    // Swap a shell with the first unknown shell of the opposite kind
    pub fn swap_first(mut self, shell: u8, known: BitArray<u8>) -> Self {
        let shell_mask = 1 << shell;
        // Mask out the shell we want to flip
        let mask = known.0 | shell_mask;
        let other_shell = if self.live_shells.get(shell) {
            // Get the first unknown blank
            (self.live_shells.0 | mask).trailing_ones()
        } else {
            // Get the first unknown live
            (self.live_shells.0 & !mask).trailing_zeros()
        };
        assert!(other_shell != 32);
        // Flip both bits
        self.assert_valid();
        self.live_shells.0 ^= shell_mask | (1 << other_shell);
        self.assert_valid();
        self
    }

    pub fn try_swap(&self, shell: u8, known: BitArray<u8>) -> Option<u8> {
        let shell_mask = 1 << shell;
        let mask = known.0 | shell_mask;
        let shell = if self.live_shells.get(shell) {
            (self.live_shells.0 | mask).trailing_ones()
        } else {
            (self.live_shells.0 & !mask).trailing_zeros()
        };
        if shell < self.n_shells as u32 {
            Some(shell as u8)
        } else {
            None
        }
    }

    pub fn all_swaps(&self, shell: u8, known: BitArray<u8>) -> BitArray<u8> {
        let shell_mask = 1 << shell;
        let mask = known.0 | shell_mask;
        if self.live_shells.get(shell) {
            BitArray(!(self.live_shells.0 | mask | !BitArray::<u8>::all(self.n_shells).0))
        } else {
            BitArray(self.live_shells.0 & !mask)
        }
    }

    pub fn all_player_known(&self) -> [BitArray<u8>; PLAYERS] {
        std::array::from_fn(|player| self.players[player].known_shells)
    }

    pub fn all_player_swaps(&self, shell: u8) -> [BitArray<u8>; PLAYERS] {
        let shell_mask = 1 << shell;
        let all_known = self.all_player_known();
        if self.live_shells.get(shell) {
            let mask = shell_mask | self.live_shells.0 | !BitArray::<u8>::all(self.n_shells).0;
            all_known.map(|known| BitArray(!(known.0 | mask)))
        } else {
            all_known.map(|known| BitArray(self.live_shells.0 & (!known.0 | shell_mask)))
        }
    }

    pub fn reveal(mut self, player: u8, shell: u8, live: bool) -> Self {
        self.players[player as usize].known_shells.set(shell, true);
        if live != self.live_shells.get(shell) {
            let known_shells = self.players[player as usize].known_shells;
            self.swap_first(shell, known_shells)
        } else {
            self
        }
    }

    pub fn visit_reveal_next<F: FnMut(Self, f32)>(
        self,
        known: BitArray<u8>,
        chance: f32,
        mut f: F,
    ) {
        if known.front() {
            f(self, chance);
        } else {
            let live_chance = self.live_chance(0, known);
            if self.live_shells.front() {
                f(self.clone(), live_chance * chance);
                f(self.swap_first(0, known), (1.0 - live_chance) * chance);
            } else {
                f(self.clone(), (1.0 - live_chance) * chance);
                f(self.swap_first(0, known), live_chance * chance);
            };
        }
    }

    pub fn end_round(mut self) -> Self {
        // In singleplayer, if the shotgun is empty, the player is not dead, then un-handcuff us and reset the turn.
        if !MULTIPLAYER && self.n_shells == 0 && self.players[self.turn as usize].health != 0 {
            self.handcuffed.set(self.turn, false);
            self.turn = 0;
            return self;
        }
        self
    }

    pub fn continue_turn(self) -> Self {
        if self.n_shells == 0 {
            return self.end_round();
        }
        self
    }

    pub fn end_turn(mut self) -> Self {
        for _ in 0..PLAYERS {
            if self.forward {
                self.turn = self.turn.saturating_add(1);
                if self.turn == PLAYERS as u8 {
                    self.turn = 0;
                }
            } else if self.turn == 0 {
                self.turn = (PLAYERS - 1) as u8;
            } else {
                self.turn -= 1;
            }
            if self.players[self.turn as usize].health == 0 {
                continue;
            } else if self.handcuffed.get(self.turn) {
                self.handcuffed.set(self.turn, false);
                continue;
            }
            break;
        }
        if self.n_shells == 0 {
            return self.end_round();
        }
        self
    }

    pub fn game_over(&self) -> bool {
        let mut players_alive = 0;
        for player in 0..PLAYERS {
            if self.players[player].health > 0 {
                players_alive += 1;
            }
        }
        players_alive < 2
    }

    pub fn round_over(&self) -> bool {
        self.game_over() || self.n_shells == 0
    }
}

pub fn fmt_percent(value: f32) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_infinite() {
        "∞".to_string()
    } else {
        format!("{:.0}%", value * 100.0)
    }
}

pub const EPSILON: f32 = 0.0001;

pub fn fmt_weight(weight: f32) -> String {
    if (weight - 1.0).abs() < EPSILON {
        return "WW".to_string();
    } else if weight.abs() < EPSILON {
        return "LL".to_string();
    }
    format!("{:0>2.0}", weight * 100.0)
    // let weight = (weight - 0.5) * 20.0 * 2.0;
    // if weight > 0.0 {
    //     format!("+{:.0}", weight)
    // } else {
    //     format!("{:.0}", weight)
    // }
}
