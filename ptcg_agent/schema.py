# Extracted from the preserved GALEX submission by tools/build_release.py.
# Inference calculations are retained; see docs/RELEASE.md and NOTICE.md.


from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from collections import defaultdict
from typing import Any
import base64, hashlib, json, math
import numpy as np

class AreaType(IntEnum):
    DECK = (1,)
    HAND = (2,)
    DISCARD = (3,)
    ACTIVE = (4,)
    BENCH = (5,)
    PRIZE = (6,)
    STADIUM = (7,)
    ENERGY = (8,)
    TOOL = (9,)
    PRE_EVOLUTION = (10,)
    PLAYER = (11,)
    LOOKING = (12,)

class EnergyType(IntEnum):
    COLORLESS = (0,)
    GRASS = (1,)
    FIRE = (2,)
    WATER = (3,)
    LIGHTNING = (4,)
    PSYCHIC = (5,)
    FIGHTING = (6,)
    DARKNESS = (7,)
    METAL = (8,)
    DRAGON = (9,)
    RAINBOW = (10,)
    TEAM_ROCKET = (11,)

class CardType(IntEnum):
    POKEMON = (0,)
    ITEM = (1,)
    TOOL = (2,)
    SUPPORTER = (3,)
    STADIUM = (4,)
    BASIC_ENERGY = (5,)
    SPECIAL_ENERGY = (6,)

class SpecialConditionType(IntEnum):
    POISON = (0,)
    BURN = (1,)
    SLEEP = (2,)
    PARALYZE = (3,)
    CONFUSE = (4,)

class SelectType(IntEnum):
    MAIN = (0,)
    CARD = (1,)
    ATTACHED_CARD = (2,)
    CARD_OR_ATTACHED_CARD = (3,)
    ENERGY = (4,)
    SKILL = (5,)
    ATTACK = (6,)
    EVOLVE = (7,)
    COUNT = (8,)
    YES_NO = (9,)
    SPECIAL_CONDITION = (10,)

class SelectContext(IntEnum):
    MAIN = (0,)
    SETUP_ACTIVE_POKEMON = (1,)
    SETUP_BENCH_POKEMON = (2,)
    SWITCH = (3,)
    TO_ACTIVE = (4,)
    TO_BENCH = (5,)
    TO_FIELD = (6,)
    TO_HAND = (7,)
    DISCARD = (8,)
    TO_DECK = (9,)
    TO_DECK_BOTTOM = (10,)
    TO_PRIZE = (11,)
    NOT_MOVE = (12,)
    DAMAGE_COUNTER = (13,)
    DAMAGE_COUNTER_ANY = (14,)
    DAMAGE = (15,)
    REMOVE_DAMAGE_COUNTER = (16,)
    HEAL = (17,)
    EVOLVES_FROM = (18,)
    EVOLVES_TO = (19,)
    DEVOLVE = (20,)
    ATTACH_FROM = (21,)
    ATTACH_TO = (22,)
    DETACH_FROM = (23,)
    LOOK = (24,)
    EFFECT_TARGET = (25,)
    DISCARD_ENERGY_CARD = (26,)
    DISCARD_TOOL_CARD = (27,)
    SWITCH_ENERGY_CARD = (28,)
    DISCARD_CARD_OR_ATTACHED_CARD = (29,)
    DISCARD_ENERGY = (30,)
    TO_HAND_ENERGY = (31,)
    TO_DECK_ENERGY = (32,)
    SWITCH_ENERGY = (33,)
    SKILL_ORDER = (34,)
    ATTACK = (35,)
    DISABLE_ATTACK = (36,)
    EVOLVE = (37,)
    DRAW_COUNT = (38,)
    DAMAGE_COUNTER_COUNT = (39,)
    REMOVE_DAMAGE_COUNTER_COUNT = (40,)
    IS_FIRST = (41,)
    MULLIGAN = (42,)
    ACTIVATE = (43,)
    FIRST_EFFECT = (44,)
    MORE_DEVOLVE = (45,)
    COIN_HEAD = (46,)
    AFFECT_SPECIAL_CONDITION = (47,)
    RECOVER_SPECIAL_CONDITION = (48,)

class OptionType(IntEnum):
    NUMBER = (0,)
    YES = (1,)
    NO = (2,)
    CARD = (3,)
    TOOL_CARD = (4,)
    ENERGY_CARD = (5,)
    ENERGY = (6,)
    PLAY = (7,)
    ATTACH = (8,)
    EVOLVE = (9,)
    ABILITY = (10,)
    DISCARD = (11,)
    RETREAT = (12,)
    ATTACK = (13,)
    END = (14,)
    SKILL = (15,)
    SPECIAL_CONDITION = (16,)

class LogType(IntEnum):
    SHUFFLE = (0,)
    HAS_BASIC_POKEMON = (1,)
    TURN_START = (2,)
    TURN_END = (3,)
    DRAW = (4,)
    DRAW_REVERSE = (5,)
    MOVE_CARD = (6,)
    MOVE_CARD_REVERSE = (7,)
    SWITCH = (8,)
    CHANGE = (9,)
    PLAY = (10,)
    ATTACH = (11,)
    EVOLVE = (12,)
    DEVOLVE = (13,)
    MOVE_ATTACHED = (14,)
    ATTACK = (15,)
    HP_CHANGE = (16,)
    POISONED = (17,)
    BURNED = (18,)
    ASLEEP = (19,)
    PARALYZED = (20,)
    CONFUSED = (21,)
    COIN = (22,)
    RESULT = (23,)

@dataclass
class Card:
    id: int
    serial: int
    playerIndex: int

@dataclass
class Pokemon:
    id: int
    serial: int
    hp: int
    maxHp: int
    appearThisTurn: bool
    energies: list[EnergyType]
    energyCards: list[Card]
    tools: list[Card]
    preEvolution: list[Card]

@dataclass
class PlayerState:
    active: list[Pokemon | None]
    bench: list[Pokemon]
    benchMax: int
    deckCount: int
    discard: list[Card]
    prize: list[Card | None]
    handCount: int
    hand: list[Card] | None
    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool

@dataclass
class State:
    turn: int
    turnActionCount: int
    yourIndex: int
    firstPlayer: int
    supporterPlayed: bool
    stadiumPlayed: bool
    energyAttached: bool
    retreated: bool
    result: int
    stadium: list[Card]
    looking: list[Card | None] | None
    players: list[PlayerState]

@dataclass
class Option:
    type: OptionType
    number: int | None = None
    area: AreaType | None = None
    index: int | None = None
    playerIndex: int | None = None
    toolIndex: int | None = None
    energyIndex: int | None = None
    count: int | None = None
    inPlayArea: AreaType | None = None
    inPlayIndex: int | None = None
    attackId: int | None = None
    cardId: int | None = None
    serial: int | None = None
    specialConditionType: SpecialConditionType | None = None

@dataclass
class SelectData:
    type: SelectType
    context: SelectContext
    minCount: int
    maxCount: int
    remainDamageCounter: int
    remainEnergyCost: int
    option: list[Option]
    deck: list[Card] | None
    contextCard: Card | None
    effect: Card | None

@dataclass
class Log:
    type: LogType
    playerIndex: int | None = None
    hasBasicPokemon: bool | None = None
    cardId: int | None = None
    serial: int | None = None
    fromArea: AreaType | None = None
    toArea: AreaType | None = None
    cardIdActive: int | None = None
    serialActive: int | None = None
    cardIdBench: int | None = None
    serialBench: int | None = None
    cardIdBefore: int | None = None
    serialBefore: int | None = None
    cardIdAfter: int | None = None
    serialAfter: int | None = None
    cardIdTarget: int | None = None
    serialTarget: int | None = None
    attackId: int | None = None
    value: int | None = None
    putDamageCounter: bool | None = None
    isRecover: bool | None = None
    head: bool | None = None
    result: int | None = None
    reason: int | None = None

@dataclass
class Observation:
    select: SelectData | None
    logs: list[Log]
    current: State | None
    search_begin_input: str | None = None

@dataclass
class SearchState:
    observation: Observation
    searchId: int

@dataclass
class ApiResult:
    state: SearchState | None
    error: int

@dataclass
class Skill:
    name: str
    text: str

@dataclass
class CardData:
    cardId: int
    name: str
    cardType: CardType
    retreatCost: int
    hp: int
    weakness: EnergyType | None
    resistance: EnergyType | None
    energyType: EnergyType
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    megaEx: bool
    tera: bool
    aceSpec: bool
    evolvesFrom: str | None
    skills: list[Skill]
    attacks: list[int]

def _dataclass_target(annotation):
    if hasattr(annotation, '__dataclass_fields__'):
        return annotation
    for candidate in getattr(annotation, '__args__', ()):
        resolved = _dataclass_target(candidate)
        if resolved is not None:
            return resolved
    return None

def _build_sdk_object(value, cls):
    if value is None:
        return None
    target = _dataclass_target(cls)
    if target is None or not isinstance(value, dict):
        return value
    fields = target.__dataclass_fields__
    converted = {}
    for key, item in value.items():
        if key not in fields:
            continue
        nested = _dataclass_target(fields[key].type)
        if isinstance(item, dict) and nested is not None:
            converted[key] = _build_sdk_object(item, nested)
        elif isinstance(item, list) and nested is not None:
            converted[key] = [_build_sdk_object(entry, nested) if isinstance(entry, dict) else entry for entry in item]
        else:
            converted[key] = item
    return target(**converted)

def _decode_game_observation(obs):
    return _build_sdk_object(obs, Observation)
