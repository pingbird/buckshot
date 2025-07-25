use num_traits::{PrimInt, WrappingSub};
use std::fmt::{Display, Formatter};

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

const fn imask(item: Item) -> u16 {
    1 << (item as u16)
}

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
    pub const ALL: ItemSet = ItemSet(0b111111111);

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

impl<T: PrimInt + WrappingSub> BitArray<T> {
    pub fn zero() -> Self {
        BitArray(T::zero())
    }

    pub fn all(len: u8) -> Self {
        BitArray((T::one() << len as usize).wrapping_sub(&T::one()))
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

pub struct GameWithKnown<const MULTIPLAYER: bool, const PLAYERS: usize> {
    pub game: Game<MULTIPLAYER, PLAYERS>,
    pub known: BitArray<u8>,
}

impl<const MULTIPLAYER: bool, const PLAYERS: usize> Display
    for GameWithKnown<MULTIPLAYER, PLAYERS>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let known = self.known;
        let game = &self.game;
        let mut header = false;
        if !known.empty() {
            write!(f, "|")?;
            for i in 0..game.n_shells {
                if known.get(i) {
                    if game.live_shells.get(i) {
                        write!(f, "#")?;
                    } else {
                        write!(f, ".")?;
                    }
                } else {
                    write!(f, " ")?;
                }
            }
            write!(f, "| ")?;
            header = true;
        }
        if MULTIPLAYER {
            if game.forward {
                write!(f, "cw ")?;
            } else {
                write!(f, "ccw ")?;
            }
            header = true;
        }
        if game.sawed_off {
            write!(f, "sawed ")?;
            header = true;
        }
        if header {
            writeln!(f, "")?;
        }

        for player in 0..PLAYERS {
            if game.turn as usize == player {
                write!(f, ">")?;
            } else {
                write!(f, " ")?;
            }
            write!(f, "{}: ", player)?;
            for h in 0..game.max_health {
                if game.players[player].health > h {
                    write!(f, "♥")?;
                } else {
                    write!(f, " ")?;
                }
            }
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
            GameWithKnown {
                game: self.clone(),
                known: BitArray::all(self.n_shells)
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItemAction {
    UseSimple(Item),
    StealSimple(u8, Item),
    StealHandcuffs(u8, u8),
    Handcuff(u8),
}

impl Display for ItemAction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ItemAction::UseSimple(item) => format!("use {}", item),
                ItemAction::StealSimple(player, item) => format!("steal {} from {}", item, player),
                ItemAction::StealHandcuffs(player, target) =>
                    format!("use {} on {} from {}", Handcuffs, target, player),
                ItemAction::Handcuff(target) => format!("use {} on {}", Handcuffs, target),
            }
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Item(ItemAction),
    Shoot(u8),
}

impl Display for Action {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Action::Item(action) => action.to_string(),
                Action::Shoot(target) => format!("shoot {}", target),
            }
        )?;
        Ok(())
    }
}

impl<const MULTIPLAYER: bool, const PLAYERS: usize> Game<MULTIPLAYER, PLAYERS> {
    pub fn with_known(self, known: BitArray<u8>) -> GameWithKnown<MULTIPLAYER, PLAYERS> {
        GameWithKnown { game: self, known }
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
        let turn_items = self.players[self.turn as usize].items.to_set();
        let allowed = turn_items.intersection(items);
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

    pub fn visit_certain_inner<F: FnMut(&[Action], &Self, bool)>(
        &self,
        actions: &mut Vec<Action>,
        f: &mut F,
    ) {
        self.visit_actions(|game, action| {
            actions.push(action.clone());
            let known_shells = self.all_known_shells();
            if game.is_action_certain(known_shells, &action) {
                game.clone()
                    .apply_action(action.clone(), |next_game, next_chance| {
                        assert_eq!(next_chance, 1.0);
                        // Stop if the round is over or the turn has changed
                        if next_game.round_over() || next_game.turn != game.turn {
                            f(actions, &next_game, true);
                            return;
                        } else {
                            next_game.visit_certain_inner(actions, f);
                        }
                    });
            } else {
                // Stop if the action does not have a certain outcome
                f(actions, game, false);
            }
            actions.pop();
        });
    }

    pub fn visit_certain<F: FnMut(&[Action], &Self, bool)>(&self, f: &mut F) {
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
            self.flip(shell, known_shells)
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
        } if self.n_shells == 1 {
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
                // If the player knows all but one shell, they now know all of them
                if self.players[player].known_shells.count() == (self.n_shells - 1) as u32 {
                    self.players[player].known_shells = BitArray::all(self.n_shells);
                }
            }
        }
        self
    }

    pub fn eject_round(mut self) -> Self {
        self.n_shells -= 1;
        self.live_shells.pop_front();
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
                    let chance = 1.0 / (n_shells - 1) as f32;
                    for i in 1..n_shells - 1 {
                        f(self.clone().reveal_round(turn, i), chance)
                    }
                    f(self.reveal_round(turn, n_shells - 1), chance)
                }
            }
            MagnifyingGlass => {
                f(self.reveal_round(turn, 1), 1.0);
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
            Adrenaline => {}
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
                    f(self.eject_round(), 1.0);
                } else {
                    f(self.eject_round().end_turn(), 1.0);
                }
            }
        }
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
            Beer => true,
            Remote => true,
            HandSaw => true,
            Handcuffs => unreachable!(),
            Adrenaline => unreachable!(),
        }
    }

    fn is_item_action_certain(&self, known_shells: BitArray<u8>, action: &ItemAction) -> bool {
        match action {
            ItemAction::UseSimple(item) => self.is_item_certain(known_shells, *item),
            ItemAction::StealSimple(_, item) => self.is_item_certain(known_shells, *item),
            ItemAction::StealHandcuffs(_, _) => true,
            ItemAction::Handcuff(_) => true,
        }
    }

    pub fn is_action_certain(&self, known_shells: BitArray<u8>, action: &Action) -> bool {
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

    pub fn action_uses_shell(&self, action: &Action) -> bool {
        match action {
            Action::Item(item_action) => match item_action {
                ItemAction::UseSimple(item) => self.item_uses_shell(*item),
                ItemAction::StealSimple(_, item) => self.item_uses_shell(*item),
                _ => false,
            },
            Action::Shoot(_) => true,
        }
    }

    pub fn visit_item_reveal_chance<F: FnMut(&Self, Option<u8>, u8, f32)>(&self, item: Item, mut f: F) -> bool {
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
            _ => false
        }
    }

    pub fn visit_action_reveal_chance<F: FnMut(&Self, Option<u8>, u8, f32)>(&self, action: &Action, mut f: F) -> bool {
        match action {
            Action::Item(item_action) => match item_action {
                ItemAction::UseSimple(item) => self.visit_item_reveal_chance(*item, f),
                ItemAction::StealSimple(_, item) => self.visit_item_reveal_chance(*item, f),
                _ => false
            },
            Action::Shoot(_) => {
                f(self, None, 0, 1.0);
                true
            }
        }
    }

    pub fn live_chance_ratio(&self, shell: u8, known: BitArray<u8>) -> (u32, u32) {
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
        (unknown_live, unknown)
    }

    pub fn live_chance(&self, shell: u8, known: BitArray<u8>) -> f32 {
        let (unknown_live, unknown) = self.live_chance_ratio(shell, known);
        unknown_live as f32 / unknown as f32
    }

    // Swap a shell with the first unknown shell of the opposite kind
    pub fn flip(mut self, shell: u8, known: BitArray<u8>) -> Self {
        let shell_mask = 1 << shell;
        // Mask out the shell we want to flip
        let mask = known.0 | shell_mask;
        let shell = if self.live_shells.get(shell) {
            // Get the first unknown blank
            (self.live_shells.0 | mask).trailing_ones()
        } else {
            // Get the first unknown live
            (self.live_shells.0 & !mask).trailing_zeros()
        };
        assert!(shell != 32);
        // Flip both bits
        self.live_shells.0 ^= shell_mask | (1 << shell);
        self
    }

    pub fn can_flip(&self, shell: u8, known: BitArray<u8>) -> bool {
        let shell_mask = 1 << shell;
        let mask = known.0 | shell_mask;
        let shell = if self.live_shells.get(shell) {
            (self.live_shells.0 | mask).trailing_ones()
        } else {
            (self.live_shells.0 & !mask).trailing_zeros()
        };
        shell != 32
    }

    pub fn reveal(mut self, player: u8, shell: u8, live: bool) -> Self {
        self.players[player as usize].known_shells.set(shell, true);
        if live != self.live_shells.get(shell) {
            let known_shells = self.players[player as usize].known_shells;
            self.flip(shell, known_shells)
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
                f(self.flip(0, known), (1.0 - live_chance) * chance);
            } else {
                f(self.clone(), (1.0 - live_chance) * chance);
                f(self.flip(0, known), live_chance * chance);
            };
        }
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
        self
    }

    pub fn game_over(&self) -> bool {
        for player in 0..PLAYERS {
            if self.players[player].health == 0 {
                return true;
            }
        }
        false
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
