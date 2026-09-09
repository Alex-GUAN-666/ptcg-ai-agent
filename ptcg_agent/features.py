# Extracted from the preserved GALEX submission by tools/build_release.py.
# Inference calculations are retained; see docs/RELEASE.md and NOTICE.md.
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any
import hashlib
import math
import numpy as np
from .schema import (
    AreaType,
    Pokemon,
)


STATE_DIM, ACTION_DIM, FEATURE_DIM = (128, 64, 192)

def _num(v):
    if v is None:
        return 0.0
    v = getattr(v, 'value', v)
    return float(v[0] if isinstance(v, tuple) else v)

def _slot(ns, v, start, width):
    return start + int.from_bytes(hashlib.blake2s(f'{ns}:{v}'.encode(), digest_size=4).digest(), 'little') % width

def _card(obs, o):
    st = obs.current
    pi = o.playerIndex if o.playerIndex is not None else st.yourIndex
    try:
        p = st.players[pi]
        pools = {AreaType.HAND: p.hand, AreaType.DISCARD: p.discard, AreaType.ACTIVE: p.active, AreaType.BENCH: p.bench, AreaType.PRIZE: p.prize, AreaType.STADIUM: st.stadium, AreaType.LOOKING: st.looking, AreaType.DECK: obs.select.deck}
        pool = pools.get(o.area)
        return pool[o.index] if pool is not None else None
    except (IndexError, TypeError):
        return None

def option_features(obs, o):
    x = np.zeros(FEATURE_DIM, np.float32)
    st = obs.current
    sel = obs.select
    me, op = (st.players[st.yourIndex], st.players[1 - st.yourIndex])
    x[:24] = [min(st.turn, 100) / 100, min(st.turnActionCount, 30) / 30, len(me.prize) / 6, len(op.prize) / 6, me.deckCount / 60, op.deckCount / 60, me.handCount / 20, op.handCount / 20, len(me.bench) / 5, len(op.bench) / 5, float(st.supporterPlayed), float(st.stadiumPlayed), float(st.energyAttached), float(st.retreated), float(st.yourIndex == st.firstPlayer), len(sel.option) / 30, _num(sel.context) / 48, _num(sel.type) / 16, sel.minCount / 10, sel.maxCount / 10, sel.remainDamageCounter / 30, sel.remainEnergyCost / 10, float(bool(st.stadium)), st.stadium[0].id / 1300 if st.stadium else 0]
    for side, p in enumerate((me, op)):
        board = [q for q in list(p.active) + list(p.bench) if q is not None]
        b = 24 + side * 16
        x[b:b + 8] = [len(board) / 6, sum((q.hp for q in board)) / 2000, sum((q.maxHp for q in board)) / 2000, sum((len(q.energies) for q in board)) / 20, sum((len(q.tools) for q in board)) / 10, sum((q.hp < q.maxHp for q in board)) / 6, sum((q.appearThisTurn for q in board)) / 6, sum((p.poisoned, p.burned, p.asleep, p.paralyzed, p.confused)) / 5]
        if p.active and p.active[0] is not None:
            q = p.active[0]
            x[b + 8:b + 16] = [q.id / 1300, q.hp / 400, q.maxHp / 400, len(q.energies) / 6, len(q.tools) / 3, len(q.preEvolution) / 3, float(q.appearThisTurn), 1]
        for q in board:
            x[_slot(f'board{side}', q.id, 64, 64)] += 0.25
        if side == 0:
            for c in p.discard:
                x[_slot('discard', c.id, 64, 64)] += 0.1
    a = STATE_DIM
    x[a:a + 16] = [_num(o.type) / 24, _num(o.area) / 16, (o.index or 0) / 12, o.playerIndex or 0, _num(o.inPlayArea) / 16, (o.inPlayIndex or 0) / 6, (o.attackId or 0) / 1600, (o.cardId or 0) / 1300, (o.number or 0) / 30, (o.count or 0) / 10, (o.energyIndex or 0) / 10, (o.toolIndex or 0) / 5, _num(o.specialConditionType) / 10, float(o.serial is not None), float(o.index is not None), 1]
    card = _card(obs, o)
    if card is not None:
        x[a + 16:a + 24] = [card.id / 1300, float(isinstance(card, Pokemon)), getattr(card, 'hp', 0) / 400, getattr(card, 'maxHp', 0) / 400, len(getattr(card, 'energies', [])) / 6, len(getattr(card, 'tools', [])) / 3, float(getattr(card, 'appearThisTurn', False)), 1]
        x[_slot('option', card.id, a + 32, 32)] += 1
    if o.cardId is not None:
        x[_slot('explicit', o.cardId, a + 32, 32)] += 1
    x[a + 24] = float(o.playerIndex == st.yourIndex)
    x[a + 25] = float(o.playerIndex == 1 - st.yourIndex)
    return x

def observation_matrix(obs):
    return np.stack([option_features(obs, o) for o in obs.select.option])
MAX_TOKENS = 48

def state_tokens(obs):
    st = obs.current
    me, op = (st.players[st.yourIndex], st.players[1 - st.yourIndex])
    pairs = []

    def add(cards, zone):
        for c in cards or []:
            if c is not None:
                pairs.append((c.id, zone))
    add(me.hand, 1)
    add(me.discard, 2)
    add(me.active, 3)
    add(me.bench, 4)
    add(st.stadium, 5)
    add(op.discard, 6)
    add(op.active, 7)
    add(op.bench, 8)
    pairs = pairs[:MAX_TOKENS]
    ids = np.zeros(MAX_TOKENS, np.int16)
    zones = np.zeros(MAX_TOKENS, np.uint8)
    for i, (cid, z) in enumerate(pairs):
        ids[i] = cid
        zones[i] = z
    return (ids, zones, len(pairs))

def option_card_id(obs, o):
    st = obs.current
    pi = o.playerIndex if o.playerIndex is not None else st.yourIndex
    try:
        p = st.players[pi]
        pools = {AreaType.HAND: p.hand, AreaType.DISCARD: p.discard, AreaType.ACTIVE: p.active, AreaType.BENCH: p.bench, AreaType.PRIZE: p.prize, AreaType.STADIUM: st.stadium, AreaType.LOOKING: st.looking, AreaType.DECK: obs.select.deck}
        pool = pools.get(o.area)
        card = pool[o.index] if pool is not None else None
        return card.id if card is not None else int(o.cardId or 0)
    except (IndexError, TypeError):
        return int(o.cardId or 0)

def encode_options(obs):
    ids, zones, count = state_tokens(obs)
    options = obs.select.option
    return (np.stack([option_features(obs, o) for o in options]), np.tile(ids, (len(options), 1)), np.tile(zones, (len(options), 1)), np.full(len(options), count, np.uint8), np.array([option_card_id(obs, o) for o in options], np.int16))
MAX_ENTITIES = 56
ENTITY_NUM_DIM = 24
STATE_NUM_DIM = 32
ACTION_NUM_DIM = 32
MAX_COUNT_CLASS = 16
OWN_ACTIVE = 0
OWN_BENCH0 = 1
OPP_ACTIVE = 6
OPP_BENCH0 = 7
STADIUM = 12
OWN_HAND0 = 13
OWN_HAND_LIMIT = 20
OWN_DISCARD0 = OWN_HAND0 + OWN_HAND_LIMIT
OWN_DISCARD_LIMIT = 11
OPP_DISCARD0 = OWN_DISCARD0 + OWN_DISCARD_LIMIT
OPP_DISCARD_LIMIT = MAX_ENTITIES - OPP_DISCARD0

def _cards(zone: Any) -> list[dict[str, Any]]:
    if not zone:
        return []
    return [card for card in zone if isinstance(card, dict)]

def _card_id(card: Any) -> int:
    if isinstance(card, dict):
        return int(card.get('id') or card.get('cardId') or 0)
    return 0

def _player(raw: dict[str, Any], index: int) -> dict[str, Any]:
    players = raw.get('current', {}).get('players', [{}, {}])
    if 0 <= index < len(players) and isinstance(players[index], dict):
        return players[index]
    return {}

def _your_index(raw: dict[str, Any]) -> int:
    return int(raw.get('current', {}).get('yourIndex', 0) or 0)

def _status_flags(player: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (float(bool(player.get('asleep'))), float(bool(player.get('burned'))), float(bool(player.get('confused'))), float(bool(player.get('paralyzed'))), float(bool(player.get('poisoned'))))

def _energy_count(card: dict[str, Any]) -> int:
    energies = card.get('energies') or []
    energy_cards = card.get('energyCards') or []
    return int(max(len(energies), len(energy_cards)))

def _fill_entity(cards: np.ndarray, zones: np.ndarray, nums: np.ndarray, slot: int, zone: int, card: dict[str, Any], own: bool, player: dict[str, Any], slot_index: int) -> None:
    if slot < 0 or slot >= MAX_ENTITIES or (not isinstance(card, dict)):
        return
    hp = float(card.get('hp') or 0)
    max_hp = float(card.get('maxHp') or hp or 0)
    energy = float(_energy_count(card))
    tools = float(len(card.get('tools') or []))
    evo = float(len(card.get('preEvolution') or []))
    damage = max(max_hp - hp, 0.0)
    asleep, burned, confused, paralyzed, poisoned = _status_flags(player)
    cards[slot] = _card_id(card)
    zones[slot] = zone
    nums[slot, 0] = 1.0
    nums[slot, 1] = 1.0 if own else -1.0
    nums[slot, 2] = 1.0 if zone in (3, 7) else 0.0
    nums[slot, 3] = 1.0 if zone in (4, 8) else 0.0
    nums[slot, 4] = 1.0 if zone == 1 else 0.0
    nums[slot, 5] = 1.0 if zone in (2, 6) else 0.0
    nums[slot, 6] = 1.0 if zone == 5 else 0.0
    nums[slot, 7] = slot_index / 20.0
    nums[slot, 8] = hp / 340.0
    nums[slot, 9] = max_hp / 340.0
    nums[slot, 10] = damage / 340.0
    nums[slot, 11] = energy / 8.0
    nums[slot, 12] = tools / 4.0
    nums[slot, 13] = evo / 3.0
    nums[slot, 14] = float(bool(card.get('appearThisTurn')))
    if zone in (3, 7):
        nums[slot, 15:20] = (asleep, burned, confused, paralyzed, poisoned)
    nums[slot, 20] = float(player.get('handCount', len(player.get('hand') or [])) or 0) / 20.0
    nums[slot, 21] = float(player.get('deckCount', 0) or 0) / 60.0
    nums[slot, 22] = float(len(player.get('prize') or [])) / 6.0
    nums[slot, 23] = math.tanh(float(card.get('serial') or 0) / 200.0)

def encode_entities(raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = raw.get('current', {})
    me_index = _your_index(raw)
    opp_index = 1 - me_index
    me = _player(raw, me_index)
    opp = _player(raw, opp_index)
    cards = np.zeros(MAX_ENTITIES, np.int16)
    zones = np.zeros(MAX_ENTITIES, np.uint8)
    nums = np.zeros((MAX_ENTITIES, ENTITY_NUM_DIM), np.float32)
    for i, card in enumerate(_cards(me.get('active'))[:1]):
        _fill_entity(cards, zones, nums, OWN_ACTIVE + i, 3, card, True, me, i)
    for i, card in enumerate(_cards(me.get('bench'))[:5]):
        _fill_entity(cards, zones, nums, OWN_BENCH0 + i, 4, card, True, me, i)
    for i, card in enumerate(_cards(opp.get('active'))[:1]):
        _fill_entity(cards, zones, nums, OPP_ACTIVE + i, 7, card, False, opp, i)
    for i, card in enumerate(_cards(opp.get('bench'))[:5]):
        _fill_entity(cards, zones, nums, OPP_BENCH0 + i, 8, card, False, opp, i)
    for i, card in enumerate(_cards(current.get('stadium'))[:1]):
        _fill_entity(cards, zones, nums, STADIUM + i, 5, card, True, me, i)
    for i, card in enumerate(_cards(me.get('hand'))[:OWN_HAND_LIMIT]):
        _fill_entity(cards, zones, nums, OWN_HAND0 + i, 1, card, True, me, i)
    for i, card in enumerate(_cards(me.get('discard'))[-OWN_DISCARD_LIMIT:]):
        _fill_entity(cards, zones, nums, OWN_DISCARD0 + i, 2, card, True, me, i)
    for i, card in enumerate(_cards(opp.get('discard'))[-OPP_DISCARD_LIMIT:]):
        _fill_entity(cards, zones, nums, OPP_DISCARD0 + i, 6, card, False, opp, i)
    return (cards, zones, nums)

def _active_stats(player: dict[str, Any]) -> tuple[float, float, float]:
    active = _cards(player.get('active'))
    if not active:
        return (0.0, 0.0, 0.0)
    card = active[0]
    hp = float(card.get('hp') or 0)
    max_hp = float(card.get('maxHp') or hp or 0)
    return (hp / 340.0, max(max_hp - hp, 0.0) / 340.0, float(_energy_count(card)) / 8.0)

def encode_state_nums(raw: dict[str, Any]) -> np.ndarray:
    current = raw.get('current', {})
    select = raw.get('select') or {}
    me = _player(raw, _your_index(raw))
    opp = _player(raw, 1 - _your_index(raw))
    state = np.zeros(STATE_NUM_DIM, np.float32)
    state[0] = float(current.get('turn', 0) or 0) / 40.0
    state[1] = float(current.get('turnActionCount', 0) or 0) / 40.0
    state[2] = float(me.get('handCount', len(me.get('hand') or [])) or 0) / 20.0
    state[3] = float(opp.get('handCount', len(opp.get('hand') or [])) or 0) / 20.0
    state[4] = float(me.get('deckCount', 0) or 0) / 60.0
    state[5] = float(opp.get('deckCount', 0) or 0) / 60.0
    state[6] = float(len(me.get('prize') or [])) / 6.0
    state[7] = float(len(opp.get('prize') or [])) / 6.0
    state[8] = float(len(_cards(me.get('bench')))) / 5.0
    state[9] = float(len(_cards(opp.get('bench')))) / 5.0
    state[10:13] = _active_stats(me)
    state[13:16] = _active_stats(opp)
    state[16] = float(bool(current.get('energyAttached')))
    state[17] = float(bool(current.get('retreated')))
    state[18] = float(bool(current.get('supporterPlayed')))
    state[19] = float(bool(current.get('stadiumPlayed')))
    state[20] = float(select.get('context', 0) or 0) / 64.0
    state[21] = float(select.get('type', 0) or 0) / 16.0
    option_count = len(select.get('option') or [])
    state[22] = float(option_count) / 32.0
    state[23] = float(select.get('minCount', 0) or 0) / 16.0
    state[24] = float(select.get('maxCount', option_count) or option_count) / 16.0
    state[25:30] = _status_flags(me)
    state[30] = float(bool(current.get('firstPlayer') == _your_index(raw)))
    state[31] = float(raw.get('step', 0) or 0) / 3000.0
    return state

def _slot_for_area(raw: dict[str, Any], player_index: int | None, area: Any, index: Any) -> int:
    me = _your_index(raw)
    if player_index is None:
        player_index = me
    try:
        area_i = int(area)
        index_i = int(index)
    except (TypeError, ValueError):
        return -1
    own = int(player_index) == me
    if area_i == 3:
        return OWN_ACTIVE if own else OPP_ACTIVE
    if area_i == 4:
        return (OWN_BENCH0 if own else OPP_BENCH0) + index_i
    if area_i == 5:
        return STADIUM
    if own and area_i == 1:
        return OWN_HAND0 + index_i
    if own and area_i == 2:
        return OWN_DISCARD0 + min(index_i, OWN_DISCARD_LIMIT - 1)
    if not own and area_i == 2:
        return OPP_DISCARD0 + min(index_i, OPP_DISCARD_LIMIT - 1)
    return -1

def option_target_slot(raw: dict[str, Any], option: dict[str, Any]) -> int:
    if 'inPlayArea' in option:
        return _slot_for_area(raw, option.get('playerIndex'), option.get('inPlayArea'), option.get('inPlayIndex'))
    return _slot_for_area(raw, option.get('playerIndex'), option.get('area'), option.get('index'))

def encode_action_nums(raw: dict[str, Any], action_cards: np.ndarray, target_slots: np.ndarray) -> np.ndarray:
    select = raw.get('select') or {}
    options = select.get('option') or []
    count = len(options)
    out = np.zeros((count, ACTION_NUM_DIM), np.float32)
    entity_cards, _, entity_nums = encode_entities(raw)
    for i, option in enumerate(options):
        slot = int(target_slots[i])
        rel = entity_nums[slot] if 0 <= slot < MAX_ENTITIES and entity_cards[slot] else None
        out[i, 0] = float(option.get('type', 0) or 0) / 16.0
        out[i, 1] = float(option.get('area', 0) or 0) / 16.0
        out[i, 2] = float(option.get('index', 0) or 0) / 32.0
        out[i, 3] = float(option.get('inPlayArea', 0) or 0) / 16.0
        out[i, 4] = float(option.get('inPlayIndex', 0) or 0) / 8.0
        out[i, 5] = float(option.get('attackId', 0) or 0) / 2048.0
        out[i, 6] = float(option.get('count', 0) or 0) / 16.0
        out[i, 7] = float(action_cards[i]) / 1268.0 if i < len(action_cards) else 0.0
        out[i, 8] = float(slot) / MAX_ENTITIES if slot >= 0 else -1.0
        out[i, 9] = float(option.get('playerIndex', _your_index(raw)) == _your_index(raw))
        out[i, 10] = float(select.get('context', 0) or 0) / 64.0
        out[i, 11] = float(select.get('type', 0) or 0) / 16.0
        out[i, 12] = float(select.get('minCount', 0) or 0) / 16.0
        out[i, 13] = float(select.get('maxCount', count) or count) / 16.0
        out[i, 14] = 1.0 if option.get('type') == 14 else 0.0
        out[i, 15] = 1.0 if 'attackId' in option else 0.0
        if rel is not None:
            out[i, 16:24] = rel[1:9]
            out[i, 24:29] = rel[10:15]
        out[i, 29] = 1.0 if slot in (OPP_ACTIVE, *range(OPP_BENCH0, OPP_BENCH0 + 5)) else 0.0
        out[i, 30] = 1.0 if slot in (OWN_ACTIVE, *range(OWN_BENCH0, OWN_BENCH0 + 5)) else 0.0
        out[i, 31] = float(i) / max(count - 1, 1)
    return out

def encode_v14(raw: dict[str, Any], old_x: np.ndarray, action_cards: np.ndarray) -> dict[str, np.ndarray]:
    entity_cards, entity_zones, entity_nums = encode_entities(raw)
    target_slots = np.asarray([option_target_slot(raw, option) for option in (raw.get('select') or {}).get('option', [])], np.int16)
    return {'state_nums': encode_state_nums(raw), 'entity_cards': entity_cards, 'entity_zones': entity_zones, 'entity_nums': entity_nums, 'action_nums': encode_action_nums(raw, action_cards, target_slots), 'target_slots': target_slots}

def legal_count_mask(minimum: int, maximum: int) -> np.ndarray:
    mask = np.zeros(MAX_COUNT_CLASS + 1, bool)
    lo = max(0, min(MAX_COUNT_CLASS, int(minimum)))
    hi = max(0, min(MAX_COUNT_CLASS, int(maximum)))
    mask[lo:hi + 1] = True
    return mask
HISTORY_DIM = 48
RESOURCE_DIM = 128
KEY_CARD_COUNT = 24
AREA_DECK = 1
AREA_HAND = 2
LOG_DRAW = 4
LOG_DRAW_REVERSE = 5
LOG_MOVE_CARD = 6
LOG_MOVE_CARD_REVERSE = 7
LOG_SWITCH = 8
LOG_PLAY = 10
LOG_ATTACH = 11
LOG_EVOLVE = 12
LOG_ATTACK = 15
LOG_DAMAGE_HEAL = 16

def _safe_int(value: Any, default: int=0) -> int:
    try:
        return int(value)
    except Exception:
        return default

def _safe_card(value: Any) -> int:
    value = _safe_int(value, 0)
    return value if value > 0 else 0

def _norm(value: float, scale: float) -> float:
    return float(max(min(value / scale, 1.0), -1.0))

@dataclass
class _PublicHistoryTracker:
    logs_seen: int = 0
    last_step: int = -1
    last_log_len: int = -1
    last_turn: int = 0
    revealed: dict[int, set[int]] = field(default_factory=lambda: {0: set(), 1: set()})
    seen_from_deck: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    visible_to_hand: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    hidden_moves: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    draws: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    plays: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    attaches: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    evolves: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    attacks: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    switches: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    damage_events: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    damage_total: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    heal_total: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_attack_id: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_attack_turn: dict[int, int] = field(default_factory=lambda: defaultdict(lambda: -99))
    last_play_card: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_attach_card: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_evolve_card: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_switch_card: dict[int, int] = field(default_factory=lambda: defaultdict(int))

def _new_history_tracker() -> _PublicHistoryTracker:
    return _PublicHistoryTracker()

def _players(raw: dict[str, Any]) -> tuple[int, int]:
    current = raw.get('current') or {}
    me = _safe_int(current.get('yourIndex', raw.get('playerIndex', 0)), 0)
    return (me, 1 - me)

def _current_turn(raw: dict[str, Any]) -> int:
    current = raw.get('current') or {}
    return _safe_int(current.get('turn', raw.get('step', 0)), 0)

def _visible_card_ids(log: dict[str, Any]) -> list[int]:
    cards: list[int] = []
    for key in ('cardId', 'cardIdActive', 'cardIdBench', 'cardIdTarget'):
        card = _safe_card(log.get(key))
        if card:
            cards.append(card)
    return cards

def update_memory(memory: _PublicHistoryTracker, raw: dict[str, Any]) -> _PublicHistoryTracker:
    step = _safe_int(raw.get('step', memory.last_step + 1), 0)
    logs = raw.get('logs') or []
    if step == memory.last_step and len(logs) == memory.last_log_len:
        return memory
    if step < memory.last_step:
        memory = _new_history_tracker()
    memory.last_step = step
    memory.last_log_len = len(logs)
    memory.last_turn = _current_turn(raw)
    for log in logs:
        if not isinstance(log, dict):
            continue
        player = _safe_int(log.get('playerIndex', 0), 0)
        log_type = _safe_int(log.get('type', -1), -1)
        for card in _visible_card_ids(log):
            memory.revealed.setdefault(player, set()).add(card)
        memory.logs_seen += 1
        if log_type in (LOG_DRAW, LOG_DRAW_REVERSE):
            memory.draws[player] += 1
        elif log_type == LOG_MOVE_CARD:
            if _safe_int(log.get('fromArea', 0), 0) == AREA_DECK:
                memory.seen_from_deck[player] += 1
            if _safe_int(log.get('toArea', 0), 0) == AREA_HAND:
                memory.visible_to_hand[player] += 1
        elif log_type == LOG_MOVE_CARD_REVERSE:
            memory.hidden_moves[player] += 1
        elif log_type == LOG_SWITCH:
            memory.switches[player] += 1
            memory.last_switch_card[player] = _safe_card(log.get('cardIdActive')) or _safe_card(log.get('cardIdBench'))
        elif log_type == LOG_PLAY:
            memory.plays[player] += 1
            memory.last_play_card[player] = _safe_card(log.get('cardId'))
        elif log_type == LOG_ATTACH:
            memory.attaches[player] += 1
            memory.last_attach_card[player] = _safe_card(log.get('cardId'))
        elif log_type == LOG_EVOLVE:
            memory.evolves[player] += 1
            memory.last_evolve_card[player] = _safe_card(log.get('cardId'))
        elif log_type == LOG_ATTACK:
            memory.attacks[player] += 1
            memory.last_attack_id[player] = _safe_int(log.get('attackId', 0), 0)
            memory.last_attack_turn[player] = memory.last_turn
        elif log_type == LOG_DAMAGE_HEAL:
            value = _safe_int(log.get('value', 0), 0)
            memory.damage_events[player] += 1
            if value < 0:
                memory.damage_total[player] += -value
            elif value > 0:
                memory.heal_total[player] += value
    return memory

def _board_summary(raw: dict[str, Any], player: int) -> tuple[int, float, float]:
    public = raw.get('public') or {}
    players = public.get('players') or []
    if player < 0 or player >= len(players) or (not isinstance(players[player], dict)):
        return (0, 0.0, 0.0)
    side = players[player]
    pokemons: list[dict[str, Any]] = []
    active = side.get('active') or side.get('activePokemon')
    if isinstance(active, dict):
        pokemons.append(active)
    bench = side.get('bench') or []
    if isinstance(bench, list):
        pokemons.extend([item for item in bench if isinstance(item, dict)])
    bench_count = max(len(pokemons) - (1 if isinstance(active, dict) else 0), 0)
    total_energy = 0.0
    max_energy = 0.0
    for pokemon in pokemons:
        energy = pokemon.get('attachedEnergy') or pokemon.get('energy') or pokemon.get('energies') or pokemon.get('energyCards') or []
        count = len(energy) if isinstance(energy, list) else _safe_int(energy, 0)
        total_energy += count
        max_energy = max(max_energy, count)
    return (bench_count, total_energy, max_energy)

def encode_history(raw: dict[str, Any], memory: _PublicHistoryTracker | None) -> np.ndarray:
    if memory is None:
        memory = _new_history_tracker()
    me, opp = _players(raw)
    turn = _current_turn(raw)
    public = raw.get('public') or {}
    players = public.get('players') or [{}, {}]
    own = players[me] if me < len(players) and isinstance(players[me], dict) else {}
    other = players[opp] if opp < len(players) and isinstance(players[opp], dict) else {}
    own_bench, own_energy, own_max_energy = _board_summary(raw, me)
    opp_bench, opp_energy, opp_max_energy = _board_summary(raw, opp)
    own_prize = float(own.get('prizeCount', own.get('remainingPrizeCount', 0)) or 0)
    opp_prize = float(other.get('prizeCount', other.get('remainingPrizeCount', 0)) or 0)
    own_deck_count = float(own.get('deckCount', 0) or 0)
    opp_deck_count = float(other.get('deckCount', 0) or 0)
    logs = raw.get('logs') or []
    values = [_norm(memory.logs_seen, 260.0), _norm(turn, 80.0), _norm(memory.last_attack_id[me], 2048.0), _norm(memory.last_attack_id[opp], 2048.0), _norm(turn - memory.last_attack_turn[me], 20.0), _norm(turn - memory.last_attack_turn[opp], 20.0), _norm(memory.last_play_card[me], 1400.0), _norm(memory.last_play_card[opp], 1400.0), _norm(memory.last_attach_card[me], 1400.0), _norm(memory.last_attach_card[opp], 1400.0), _norm(memory.last_evolve_card[me], 1400.0), _norm(memory.last_evolve_card[opp], 1400.0), _norm(memory.last_switch_card[me], 1400.0), _norm(memory.last_switch_card[opp], 1400.0), _norm(len(memory.revealed.get(me, set())), 60.0), _norm(len(memory.revealed.get(opp, set())), 60.0), _norm(memory.seen_from_deck[me], 30.0), _norm(memory.seen_from_deck[opp], 30.0), _norm(memory.attacks[me], 20.0), _norm(memory.attacks[opp], 20.0), _norm(memory.plays[me], 40.0), _norm(memory.plays[opp], 40.0), _norm(memory.attaches[me], 25.0), _norm(memory.attaches[opp], 25.0), _norm(memory.evolves[me], 20.0), _norm(memory.evolves[opp], 20.0), _norm(memory.switches[me], 20.0), _norm(memory.switches[opp], 20.0), _norm(memory.damage_events[me], 50.0), _norm(memory.damage_events[opp], 50.0), _norm(memory.damage_total[me], 1200.0), _norm(memory.damage_total[opp], 1200.0), _norm(memory.heal_total[me], 800.0), _norm(memory.heal_total[opp], 800.0), _norm(memory.visible_to_hand[me], 30.0), _norm(memory.visible_to_hand[opp], 30.0), _norm(memory.hidden_moves[me], 40.0), _norm(memory.hidden_moves[opp], 40.0), _norm(own_deck_count, 60.0), _norm(opp_deck_count, 60.0), _norm(opp_prize - own_prize, 6.0), _norm(own_energy, 12.0), _norm(own_max_energy, 6.0), _norm(opp_energy, 12.0), _norm(opp_max_energy, 6.0), _norm(own_bench, 5.0), _norm(opp_bench, 5.0), _norm(len(logs), 48.0)]
    out = np.zeros(HISTORY_DIM, np.float32)
    out[:len(values)] = values
    return out

def _card_id(item: Any) -> int:
    if isinstance(item, dict):
        return _safe_card(item.get('id', item.get('cardId')))
    return 0

def _walk_cards(obj: Any) -> list[int]:
    found: list[int] = []
    if isinstance(obj, dict):
        card = _card_id(obj)
        if card:
            found.append(card)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                found.extend(_walk_cards(value))
    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, (dict, list)):
                found.extend(_walk_cards(value))
    return found

def _zone_cards(side: dict[str, Any], zone: str) -> list[int]:
    return _walk_cards(side.get(zone) or [])

def _key_cards(decklist: list[int] | None) -> list[int]:
    if not decklist:
        return [0] * KEY_CARD_COUNT
    counts: dict[int, int] = {}
    for card in decklist:
        card = int(card)
        counts[card] = counts.get(card, 0) + 1
    ordered = sorted(counts, key=lambda card: (-counts[card], card))
    return (ordered + [0] * KEY_CARD_COUNT)[:KEY_CARD_COUNT]

def _count(ids: list[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for card in ids:
        if card:
            out[card] = out.get(card, 0) + 1
    return out

def _side(raw: dict[str, Any], player: int) -> dict[str, Any]:
    current = raw.get('current') or {}
    public = raw.get('public') or {}
    players = current.get('players') or public.get('players') or []
    if 0 <= player < len(players) and isinstance(players[player], dict):
        return players[player]
    return {}

def encode_resources(raw: dict[str, Any], memory: _PublicHistoryTracker | None, decklist: list[int] | None) -> np.ndarray:
    if memory is None:
        memory = _new_history_tracker()
    me, opp = _players(raw)
    own = _side(raw, me)
    other = _side(raw, opp)
    deck_counts = _count([int(x) for x in decklist or []])
    keys = _key_cards(decklist)
    hand_counts = _count(_zone_cards(own, 'hand'))
    discard_counts = _count(_zone_cards(own, 'discard'))
    active_counts = _count(_zone_cards(own, 'active'))
    bench_counts = _count(_zone_cards(own, 'bench'))
    visible_counts: dict[int, int] = {}
    for source in (hand_counts, discard_counts, active_counts, bench_counts):
        for card, value in source.items():
            visible_counts[card] = visible_counts.get(card, 0) + value
    for card in memory.revealed.get(me, set()):
        if card in deck_counts:
            visible_counts[card] = max(visible_counts.get(card, 0), 1)
    own_deck_count = float(own.get('deckCount', 0) or 0)
    own_prize_unknown = sum((1 for item in own.get('prize') or [] if item is None))
    unknown_total = max(own_deck_count + own_prize_unknown, 1.0)
    deck_probability = own_deck_count / unknown_total
    out = np.zeros(RESOURCE_DIM, np.float32)
    cursor = 0
    for card in keys:
        total = deck_counts.get(card, 0)
        hand = hand_counts.get(card, 0)
        visible = visible_counts.get(card, 0)
        unknown = max(total - visible, 0)
        expected_deck = unknown * deck_probability
        out[cursor:cursor + 4] = (_norm(total, 8.0), _norm(hand, 4.0), _norm(visible, 8.0), _norm(expected_deck, 8.0))
        cursor += 4
    own_bench, own_energy, own_max_energy = _board_summary(raw, me)
    opp_bench, opp_energy, opp_max_energy = _board_summary(raw, opp)
    active_cards = [item for item in own.get('active') or [] if isinstance(item, dict)]
    opp_active_cards = [item for item in other.get('active') or [] if isinstance(item, dict)]
    active = active_cards[0] if active_cards else {}
    opp_active = opp_active_cards[0] if opp_active_cards else {}
    select = raw.get('select') or {}
    energy_like = {card for card, total in deck_counts.items() if total > 4}
    hand_energy = sum((value for card, value in hand_counts.items() if card in energy_like))
    discard_energy = sum((value for card, value in discard_counts.items() if card in energy_like))
    low_hp_own = 0
    low_hp_opp = 0
    for card_obj in [x for x in (own.get('active') or []) + (own.get('bench') or []) if isinstance(x, dict)]:
        hp = float(card_obj.get('hp', 0) or 0)
        max_hp = float(card_obj.get('maxHp', hp) or hp or 1)
        low_hp_own += hp / max_hp <= 0.35
    for card_obj in [x for x in (other.get('active') or []) + (other.get('bench') or []) if isinstance(x, dict)]:
        hp = float(card_obj.get('hp', 0) or 0)
        max_hp = float(card_obj.get('maxHp', hp) or hp or 1)
        low_hp_opp += hp / max_hp <= 0.35
    summary = [_norm(own.get('handCount', len(own.get('hand') or [])) or 0, 20.0), _norm(own_deck_count, 60.0), _norm(len(own.get('discard') or []), 30.0), _norm(own_prize_unknown, 6.0), _norm(hand_energy, 10.0), _norm(discard_energy, 15.0), _norm(own_bench, 5.0), _norm(opp_bench, 5.0), _norm(own_energy, 14.0), _norm(own_max_energy, 6.0), _norm(opp_energy, 14.0), _norm(opp_max_energy, 6.0), _norm(float(active.get('hp', 0) or 0), 250.0), _norm(float(active.get('maxHp', 0) or 0), 250.0), _norm(float(opp_active.get('hp', 0) or 0), 250.0), _norm(float(opp_active.get('maxHp', 0) or 0), 250.0), _norm(low_hp_own, 6.0), _norm(low_hp_opp, 6.0), _norm(int(bool(raw.get('current', {}).get('energyAttached', False))), 1.0), _norm(int(bool(raw.get('current', {}).get('supporterPlayed', False))), 1.0), _norm(int(bool(raw.get('current', {}).get('stadiumPlayed', False))), 1.0), _norm(int(bool(raw.get('current', {}).get('retreated', False))), 1.0), _norm(select.get('type', 0) or 0, 64.0), _norm(select.get('context', 0) or 0, 128.0), _norm(select.get('minCount', 0) or 0, 8.0), _norm(select.get('maxCount', 0) or 0, 8.0), _norm(len(select.get('option') or []), 64.0), _norm(select.get('remainDamageCounter', 0) or 0, 300.0), _norm(select.get('remainEnergyCost', 0) or 0, 6.0), _norm(memory.visible_to_hand[me], 30.0), _norm(memory.visible_to_hand[opp], 30.0), _norm(memory.seen_from_deck[opp], 30.0)]
    room = max(0, RESOURCE_DIM - cursor)
    out[cursor:cursor + min(len(summary), room)] = summary[:room]
    return out
TACTIC_DIM = 96
FEATURE_VARIANTS = {'core_v36': ((0, 48),), 'card': ((0, 72),), 'attack': ((0, 48), (72, 88)), 'threat': ((0, 48), (88, 96)), 'card_attack': ((0, 88),), 'card_threat': ((0, 72), (88, 96)), 'attack_threat': ((0, 48), (72, 96)), 'full': ((0, 96),)}

def feature_mask_for(name: str, width: int=TACTIC_DIM) -> np.ndarray:
    mask = np.zeros(width, np.float32)
    for start, end in FEATURE_VARIANTS.get(name, FEATURE_VARIANTS['full']):
        mask[max(0, start):min(width, end)] = 1.0
    return mask
_STATIC_READY = False
_CARD_TYPE = None
_CARD_ENERGY = None
_CARD_HP = None
_CARD_RETREAT = None
_CARD_WEAKNESS = None
_CARD_RESISTANCE = None
_CARD_STAGE = None
_CARD_FLAGS = None
_ATTACK_DAMAGE = None
_ATTACK_ENERGY_COUNT = None
_ATTACK_ENERGY_TYPES = None
_CARD_ATTACKS = None
_CARD_FUNC_FLAGS = None
_ATTACK_EFFECT_FLAGS = None
_CARD_FUNCTION_PATTERNS = (('draw', ('draw', 'until you have', 'cards in your hand')), ('search', ('search your deck', 'look through your deck')), ('pokemon_search', ('search your deck for', 'pokemon')), ('energy_search', ('energy card', 'basic energy', 'attach an energy')), ('attach_energy', ('attach', 'energy', 'from your discard', 'from your hand')), ('discard', ('discard', 'put into your discard')), ('discard_energy', ('discard an energy', 'discard all energy', 'discard energy')), ('switch', ('switch', 'switch your active', 'switch this pokemon')), ('gust', ('switch 1 of your opponent', "opponent's benched", 'opponent’s benched')), ('heal', ('heal', 'remove damage')), ('damage_counter', ('damage counter', 'put damage counters')), ('evolve', ('evolve', 'evolution', 'evolves from')), ('bench', ('bench', 'benched pokemon')), ('hand_disrupt', ("opponent's hand", 'opponent’s hand', 'shuffle their hand')), ('deck_disrupt', ('discard cards from', 'top of your opponent', "opponent's deck")), ('recover', ('from your discard pile', 'put into your hand', 'shuffle into your deck')), ('prevent', ('prevent', 'take less damage', "isn't affected", 'is not affected')), ('condition', ('poisoned', 'burned', 'asleep', 'paralyzed', 'confused')), ('coin', ('flip a coin', 'flip coins')), ('retreat', ('retreat', 'retreat cost')), ('supporter_like', ('supporter',)), ('item_like', ('item',)), ('stadium_like', ('stadium',)), ('tool_like', ('tool', 'pokemon tool', 'pokémon tool')))
_ATTACK_EFFECT_PATTERNS = (('search', ('search your deck', 'look through your deck')), ('draw', ('draw',)), ('bench_damage', ('benched', 'bench')), ('active_damage', ('active pokemon', 'active pokémon')), ('discard_energy', ('discard an energy', 'discard all energy', 'discard energy')), ('attach_energy', ('attach', 'energy')), ('heal', ('heal', 'remove damage')), ('switch', ('switch',)), ('gust', ("opponent's benched", 'opponent’s benched', 'switch 1 of your opponent')), ('coin', ('flip a coin', 'flip coins')), ('poison', ('poisoned',)), ('burn', ('burned',)), ('sleep', ('asleep',)), ('paralyze', ('paralyzed',)), ('confuse', ('confused',)), ('cant_attack', ("can't attack", 'cannot attack')), ('prevent', ('prevent', "isn't affected", 'is not affected')), ('self_damage', ('does damage to itself', 'this pokemon does', 'this pokémon does')), ('conditional_damage', ('more damage', 'for each', 'times', 'if your opponent', 'if this pokemon')), ('shuffle', ('shuffle',)), ('discard_deck', ('top of your opponent', 'discard cards from')), ('hand_disrupt', ("opponent's hand", 'opponent’s hand')), ('retreat', ('retreat',)), ('energy_scaling', ('energy attached', 'amount of energy')))

def _text_of_card(card: Any, attacks_by_card: list[list[int]] | None=None) -> str:
    parts = [str(getattr(card, 'name', '') or '')]
    for skill in getattr(card, 'skills', []) or []:
        parts.append(str(getattr(skill, 'name', '') or ''))
        parts.append(str(getattr(skill, 'text', '') or ''))
    return '\n'.join(parts).lower()

def _pattern_flags(text: str, patterns: tuple[tuple[str, tuple[str, ...]], ...]) -> np.ndarray:
    out = np.zeros(len(patterns), np.float32)
    lower = (text or '').lower()
    for i, (_, keys) in enumerate(patterns):
        if any((key in lower for key in keys)):
            out[i] = 1.0
    return out

def _build_card_function_flags(max_card: int, attacks_by_card: list[list[int]]) -> np.ndarray:
    flags = np.zeros((max_card, len(_CARD_FUNCTION_PATTERNS)), np.float32)
    try:
        from cg.api import all_card_data, all_attack
        attack_text = {int(getattr(a, 'attackId', 0) or 0): (str(getattr(a, 'name', '') or '') + '\n' + str(getattr(a, 'text', '') or '')).lower() for a in all_attack()}
        for card in all_card_data():
            cid = int(getattr(card, 'cardId', 0) or 0)
            if not 0 <= cid < max_card:
                continue
            text = _text_of_card(card)
            for attack_id in attacks_by_card[cid] if cid < len(attacks_by_card) else []:
                text += '\n' + attack_text.get(int(attack_id), '')
            row = _pattern_flags(text, _CARD_FUNCTION_PATTERNS)
            ctype = int(getattr(card, 'cardType', 0) or 0)
            if getattr(card, 'hp', 0):
                row[12] = max(row[12], 1.0)
            if getattr(card, 'energyType', None) is not None and ctype in (5, 6):
                row[3] = max(row[3], 1.0)
                row[4] = max(row[4], 1.0)
            flags[cid, :len(row)] = row
    except Exception:
        pass
    return flags

def _build_attack_effect_flags(max_attack: int) -> np.ndarray:
    flags = np.zeros((max_attack, len(_ATTACK_EFFECT_PATTERNS)), np.float32)
    try:
        from cg.api import all_attack
        for attack in all_attack():
            aid = int(getattr(attack, 'attackId', 0) or 0)
            if not 0 <= aid < max_attack:
                continue
            text = (str(getattr(attack, 'name', '') or '') + '\n' + str(getattr(attack, 'text', '') or '')).lower()
            flags[aid, :] = _pattern_flags(text, _ATTACK_EFFECT_PATTERNS)
            if float(getattr(attack, 'damage', 0) or 0) > 0:
                flags[aid, 3] = max(flags[aid, 3], 1.0)
    except Exception:
        pass
    return flags

def _ensure_static_tables() -> None:
    global _STATIC_READY, _CARD_TYPE, _CARD_ENERGY, _CARD_HP, _CARD_RETREAT, _CARD_WEAKNESS, _CARD_RESISTANCE, _CARD_STAGE, _CARD_FLAGS, _ATTACK_DAMAGE, _ATTACK_ENERGY_COUNT, _ATTACK_ENERGY_TYPES, _CARD_ATTACKS, _CARD_FUNC_FLAGS, _ATTACK_EFFECT_FLAGS
    if _STATIC_READY:
        return
    max_card = 2048
    max_attack = 4096
    _CARD_TYPE = np.zeros(max_card, np.float32)
    _CARD_ENERGY = np.full(max_card, -1, np.float32)
    _CARD_HP = np.zeros(max_card, np.float32)
    _CARD_RETREAT = np.zeros(max_card, np.float32)
    _CARD_WEAKNESS = np.full(max_card, -1, np.float32)
    _CARD_RESISTANCE = np.full(max_card, -1, np.float32)
    _CARD_STAGE = np.zeros((max_card, 3), np.float32)
    _CARD_FLAGS = np.zeros((max_card, 4), np.float32)
    _ATTACK_DAMAGE = np.zeros(max_attack, np.float32)
    _ATTACK_ENERGY_COUNT = np.zeros(max_attack, np.float32)
    _ATTACK_ENERGY_TYPES = np.zeros((max_attack, 12), np.float32)
    attacks_by_card: list[list[int]] = [[] for _ in range(max_card)]
    try:
        from cg.api import all_card_data, all_attack
        for card in all_card_data():
            cid = int(getattr(card, 'cardId', 0) or 0)
            if not 0 <= cid < max_card:
                continue
            _CARD_TYPE[cid] = float(getattr(card, 'cardType', 0) or 0) / 8.0
            energy_type = getattr(card, 'energyType', None)
            if energy_type is not None:
                _CARD_ENERGY[cid] = float(int(energy_type))
            _CARD_HP[cid] = _norm(float(getattr(card, 'hp', 0) or 0), 400.0)
            _CARD_RETREAT[cid] = _norm(float(getattr(card, 'retreatCost', 0) or 0), 5.0)
            weakness = getattr(card, 'weakness', None)
            resistance = getattr(card, 'resistance', None)
            if weakness is not None:
                _CARD_WEAKNESS[cid] = float(int(weakness))
            if resistance is not None:
                _CARD_RESISTANCE[cid] = float(int(resistance))
            _CARD_STAGE[cid, 0] = float(bool(getattr(card, 'basic', False)))
            _CARD_STAGE[cid, 1] = float(bool(getattr(card, 'stage1', False)))
            _CARD_STAGE[cid, 2] = float(bool(getattr(card, 'stage2', False)))
            _CARD_FLAGS[cid, 0] = float(bool(getattr(card, 'ex', False)))
            _CARD_FLAGS[cid, 1] = float(bool(getattr(card, 'megaEx', False)))
            _CARD_FLAGS[cid, 2] = float(bool(getattr(card, 'tera', False)))
            _CARD_FLAGS[cid, 3] = float(bool(getattr(card, 'aceSpec', False)))
            card_attacks = getattr(card, 'attacks', [])
            if isinstance(card_attacks, list):
                attacks_by_card[cid] = [int(a) for a in card_attacks if 0 <= int(a) < max_attack]
        for attack in all_attack():
            aid = int(getattr(attack, 'attackId', 0) or 0)
            if not 0 <= aid < max_attack:
                continue
            _ATTACK_DAMAGE[aid] = _norm(float(getattr(attack, 'damage', 0) or 0), 300.0)
            energies = getattr(attack, 'energies', [])
            if isinstance(energies, list):
                _ATTACK_ENERGY_COUNT[aid] = _norm(float(len(energies)), 5.0)
                for energy in energies:
                    et = int(energy)
                    if 0 <= et < 12:
                        _ATTACK_ENERGY_TYPES[aid, et] += 1.0 / 5.0
    except Exception:
        pass
    _CARD_ATTACKS = tuple((tuple(x) for x in attacks_by_card))
    _CARD_FUNC_FLAGS = _build_card_function_flags(max_card, attacks_by_card)
    _ATTACK_EFFECT_FLAGS = _build_attack_effect_flags(max_attack)
    _STATIC_READY = True

def _active_card(side: dict[str, Any]) -> dict[str, Any]:
    active = side.get('active') or []
    return active[0] if isinstance(active, list) and active and isinstance(active[0], dict) else {}

def _card_static(card_id: int) -> np.ndarray:
    _ensure_static_tables()
    out = np.zeros(16, np.float32)
    cid = int(card_id)
    if not 0 <= cid < len(_CARD_TYPE):
        return out
    out[0] = _CARD_TYPE[cid]
    out[1] = _norm(_CARD_ENERGY[cid], 12.0) if _CARD_ENERGY[cid] >= 0 else -1.0
    out[2] = _CARD_HP[cid]
    out[3] = _CARD_RETREAT[cid]
    out[4] = _norm(_CARD_WEAKNESS[cid], 12.0) if _CARD_WEAKNESS[cid] >= 0 else -1.0
    out[5] = _norm(_CARD_RESISTANCE[cid], 12.0) if _CARD_RESISTANCE[cid] >= 0 else -1.0
    out[6:9] = _CARD_STAGE[cid]
    out[9:13] = _CARD_FLAGS[cid]
    attacks = _CARD_ATTACKS[cid] if _CARD_ATTACKS is not None and cid < len(_CARD_ATTACKS) else ()
    if attacks:
        damages = [_ATTACK_DAMAGE[a] for a in attacks if 0 <= a < len(_ATTACK_DAMAGE)]
        out[13] = max(damages) if damages else 0.0
        out[14] = _norm(float(len(attacks)), 4.0)
    out[15] = float(cid > 0)
    return out

def _attack_static(attack_id: int) -> np.ndarray:
    _ensure_static_tables()
    out = np.zeros(16, np.float32)
    aid = int(attack_id)
    if 0 <= aid < len(_ATTACK_DAMAGE):
        out[0] = _ATTACK_DAMAGE[aid]
        out[1] = _ATTACK_ENERGY_COUNT[aid]
        out[2:14] = _ATTACK_ENERGY_TYPES[aid]
        out[14] = float(aid > 0)
    return out

def _card_function(card_id: int) -> np.ndarray:
    _ensure_static_tables()
    out = np.zeros(24, np.float32)
    cid = int(card_id)
    if _CARD_FUNC_FLAGS is not None and 0 <= cid < len(_CARD_FUNC_FLAGS):
        width = min(out.shape[0], _CARD_FUNC_FLAGS.shape[1])
        out[:width] = _CARD_FUNC_FLAGS[cid, :width]
    return out

def _attack_effect(attack_id: int) -> np.ndarray:
    _ensure_static_tables()
    out = np.zeros(24, np.float32)
    aid = int(attack_id)
    if _ATTACK_EFFECT_FLAGS is not None and 0 <= aid < len(_ATTACK_EFFECT_FLAGS):
        width = min(out.shape[0], _ATTACK_EFFECT_FLAGS.shape[1])
        out[:width] = _ATTACK_EFFECT_FLAGS[aid, :width]
    return out

def _hp_fields(pokemon: dict[str, Any]) -> tuple[float, float, float]:
    hp = float(pokemon.get('hp', 0) or 0)
    max_hp = float(pokemon.get('maxHp', pokemon.get('maxHP', hp)) or hp or 1.0)
    damage = max(max_hp - hp, 0.0)
    return (hp, max_hp, damage)

def _prize_count(side: dict[str, Any]) -> float:
    return float(side.get('prizeCount', side.get('remainingPrizeCount', 0)) or 0)

def encode_tactics(raw: dict[str, Any], action_cards: np.ndarray, target_slots: np.ndarray) -> np.ndarray:
    _ensure_static_tables()
    select = raw.get('select') or {}
    options = select.get('option') or []
    count = len(options)
    out = np.zeros((count, TACTIC_DIM), np.float32)
    me, opp = _players(raw)
    own = _side(raw, me)
    other = _side(raw, opp)
    own_active = _active_card(own)
    opp_active = _active_card(other)
    own_active_id = _card_id(own_active)
    opp_active_id = _card_id(opp_active)
    own_active_hp, own_active_max_hp, own_active_damage = _hp_fields(own_active)
    opp_active_hp, opp_active_max_hp, opp_active_damage = _hp_fields(opp_active)
    own_prize = _prize_count(own)
    opp_prize = _prize_count(other)
    own_bench, own_energy, own_max_energy = _board_summary(raw, me)
    opp_bench, opp_energy, opp_max_energy = _board_summary(raw, opp)
    own_hand = float(own.get('handCount', len(own.get('hand') or [])) or 0)
    opp_hand = float(other.get('handCount', len(other.get('hand') or [])) or 0)
    own_deck = float(own.get('deckCount', 0) or 0)
    opp_deck = float(other.get('deckCount', 0) or 0)
    entity = encode_v14(raw, np.zeros((max(count, 1), 192), np.float32), action_cards)
    entity_cards = entity['entity_cards']
    entity_nums = entity['entity_nums']
    own_type = _CARD_ENERGY[own_active_id] if 0 <= own_active_id < len(_CARD_ENERGY) else -1
    for i, option in enumerate(options):
        card_id = int(action_cards[i]) if i < len(action_cards) else 0
        attack_id = _safe_int(option.get('attackId', option.get('id', 0)), 0)
        target_slot = int(target_slots[i]) if i < len(target_slots) else -1
        target_id = int(entity_cards[target_slot]) if 0 <= target_slot < len(entity_cards) else 0
        if target_id <= 0:
            target_id = opp_active_id if attack_id else target_id
        target_hp_norm = float(entity_nums[target_slot, 8]) if 0 <= target_slot < len(entity_nums) else _norm(opp_active_hp, 340.0)
        target_damage_norm = float(entity_nums[target_slot, 10]) if 0 <= target_slot < len(entity_nums) else _norm(opp_active_damage, 340.0)
        target_hp = target_hp_norm * 340.0
        target_damage = target_damage_norm * 340.0
        target_max_hp = max(target_hp + target_damage, 1.0)
        out[i, 0:16] = _card_static(card_id)
        out[i, 16:32] = _card_static(target_id)
        out[i, 32:48] = _attack_static(attack_id)
        out[i, 48:72] = _card_function(card_id)
        out[i, 72:88] = _attack_effect(attack_id)[:16]
        damage = float(_ATTACK_DAMAGE[attack_id] * 300.0) if 0 <= attack_id < len(_ATTACK_DAMAGE) else 0.0
        target_weakness = _CARD_WEAKNESS[target_id] if 0 <= target_id < len(_CARD_WEAKNESS) else -1
        target_resistance = _CARD_RESISTANCE[target_id] if 0 <= target_id < len(_CARD_RESISTANCE) else -1
        super_effective = float(attack_id > 0 and own_type >= 0 and (target_weakness == own_type))
        resisted = float(attack_id > 0 and own_type >= 0 and (target_resistance == own_type))
        estimated_damage = max(0.0, damage * (2.0 if super_effective else 1.0) - (30.0 if resisted else 0.0))
        remaining_hp = max(target_hp - estimated_damage, 0.0)
        ko = float(attack_id > 0 and estimated_damage >= max(target_hp, 1.0))
        target_is_active = float(target_id > 0 and target_id == opp_active_id)
        target_is_damaged = float(target_damage > 0)
        likely_prize_finish = float(ko and own_prize <= (2.0 if _CARD_FLAGS[target_id, 0] > 0.5 else 1.0)) if 0 <= target_id < len(_CARD_FLAGS) else 0.0
        late_game = float(max(own_prize, opp_prize) <= 2.0)
        out[i, 88:96] = (_norm(target_hp, 340.0), _norm(remaining_hp, 340.0), ko, super_effective, resisted, _norm(estimated_damage, 300.0), target_is_active, likely_prize_finish)
    return out

def encode_v40(raw: dict[str, Any], old_x: np.ndarray, action_cards: np.ndarray, memory: _PublicHistoryTracker | None, decklist: list[int] | None=None) -> dict[str, np.ndarray]:
    extra = encode_v14(raw, old_x, action_cards)
    extra['history_nums'] = encode_history(raw, memory)
    extra['resource_nums'] = encode_resources(raw, memory, decklist)
    extra['tactic_nums'] = encode_tactics(raw, action_cards, extra['target_slots'])
    return extra

def encode_v36(raw: dict[str, Any], old_x: np.ndarray, action_cards: np.ndarray, memory: _PublicHistoryTracker | None, decklist: list[int] | None=None) -> dict[str, np.ndarray]:
    return encode_v40(raw, old_x, action_cards, memory, decklist)

def encode_v20(raw: dict[str, Any], old_x: np.ndarray, action_cards: np.ndarray, memory: _PublicHistoryTracker | None, decklist: list[int] | None=None) -> dict[str, np.ndarray]:
    return encode_v40(raw, old_x, action_cards, memory, decklist)

def encode_v18(raw: dict[str, Any], old_x: np.ndarray, action_cards: np.ndarray, memory: _PublicHistoryTracker | None) -> dict[str, np.ndarray]:
    return encode_v40(raw, old_x, action_cards, memory, None)
