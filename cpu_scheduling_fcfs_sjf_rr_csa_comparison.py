"""
CPU Scheduling Evaluator
Algorithms:
    1. FCFS
    2. Non-preemptive SJF
    3. Round Robin with q = 3
    4. Cyclic Scheduling Algorithm (CSA)

Input format:
    processes = [
        [arrival_time, burst_time],   # P0
        [arrival_time, burst_time],   # P1
        ...
    ]

Main function:
    evaluate_schedulers(processes)

Return format:
    {
        "fcfs": {...},
        "sjf": {...},
        "rr_q3": {...},
        "csa": {...}
    }

Each algorithm returns:
    - execution_sequence
    - avg_tat
    - avg_response_time
    - avg_waiting_time
    - per_process

CSA additionally returns:
    - state_table
    - tie_events

Metric definitions:
    Response Time   = First Start Time - Arrival Time
    Turnaround Time = Completion Time - Arrival Time
    Waiting Time    = Turnaround Time - Burst Time

Paper-specified CSA rules:
    - Odd CNT: choose minimum remaining time from L.
    - Even CNT: choose arrival time closest to the midpoint of the
      minimum and maximum arrival times in L.
    - QNEW = min(RTX, (Max-Min)/2, sum(RT)/N), after removing
      zero-valued terms and enforcing QNEW >= 1.

Explicit clarifications used by this implementation:
    1. Exact Select_Process ties choose the oldest process (earliest
       arrival); if arrival is also tied, choose the lower PID.
    2. Fractional QNEW values are retained; no floor/ceil rounding.
    3. Calculate_Jump continues scanning at the same queue index after
       removing an entry, so a shifted qualifying entry is not skipped.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union


Number = Union[int, float]
EPS = 1e-10


@dataclass
class Process:
    pid: int
    arrival: float
    burst: float
    remaining: float
    first_start: Optional[float] = None
    completion: Optional[float] = None
    admitted: bool = False


# ---------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------

def _validate_input(
    processes: Sequence[Sequence[Number]],
) -> List[Tuple[float, float]]:
    if not isinstance(processes, (list, tuple)) or not processes:
        raise ValueError(
            "processes must be a non-empty list like [[AT, BT], ...]"
        )

    clean: List[Tuple[float, float]] = []

    for i, item in enumerate(processes):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"P{i}: expected [arrival_time, burst_time]")

        arrival = float(item[0])
        burst = float(item[1])

        if arrival < 0:
            raise ValueError(f"P{i}: arrival time must be >= 0")
        if burst <= 0:
            raise ValueError(f"P{i}: burst time must be > 0")

        clean.append((arrival, burst))

    return clean


def _make_processes(
    data: Sequence[Tuple[float, float]],
) -> List[Process]:
    return [
        Process(
            pid=i,
            arrival=arrival,
            burst=burst,
            remaining=burst,
        )
        for i, (arrival, burst) in enumerate(data)
    ]


def _segment(
    pid: Union[int, str],
    start: float,
    end: float,
) -> Dict[str, object]:
    name = pid if isinstance(pid, str) else f"P{pid}"

    return {
        "process": name,
        "start": start,
        "end": end,
        "duration": end - start,
    }


def _build_result(
    processes: List[Process],
    timeline: List[Dict[str, object]],
) -> Dict[str, object]:
    per_process: Dict[str, Dict[str, float]] = {}

    for p in processes:
        if p.first_start is None or p.completion is None:
            raise RuntimeError(f"P{p.pid} did not complete")

        response_time = p.first_start - p.arrival
        turnaround_time = p.completion - p.arrival
        waiting_time = turnaround_time - p.burst

        per_process[f"P{p.pid}"] = {
            "arrival_time": p.arrival,
            "burst_time": p.burst,
            "first_start_time": p.first_start,
            "completion_time": p.completion,
            "response_time": response_time,
            "turnaround_time": turnaround_time,
            "waiting_time": waiting_time,
        }

    n = len(processes)

    return {
        "execution_sequence": timeline,
        "avg_tat": sum(
            x["turnaround_time"] for x in per_process.values()
        ) / n,
        "avg_response_time": sum(
            x["response_time"] for x in per_process.values()
        ) / n,
        "avg_waiting_time": sum(
            x["waiting_time"] for x in per_process.values()
        ) / n,
        "per_process": per_process,
    }


# ---------------------------------------------------------------------
# FCFS
# ---------------------------------------------------------------------

def _fcfs(
    data: Sequence[Tuple[float, float]],
) -> Dict[str, object]:
    ps = _make_processes(data)
    timeline: List[Dict[str, object]] = []

    t = 0.0

    order = sorted(
        range(len(ps)),
        key=lambda i: (ps[i].arrival, ps[i].pid),
    )

    for i in order:
        p = ps[i]

        if t < p.arrival - EPS:
            timeline.append(_segment("IDLE", t, p.arrival))
            t = p.arrival

        p.first_start = t

        end = t + p.burst
        timeline.append(_segment(i, t, end))

        t = end
        p.remaining = 0.0
        p.completion = t

    return _build_result(ps, timeline)


# ---------------------------------------------------------------------
# Non-preemptive SJF
# ---------------------------------------------------------------------

def _sjf(
    data: Sequence[Tuple[float, float]],
) -> Dict[str, object]:
    ps = _make_processes(data)
    timeline: List[Dict[str, object]] = []

    completed = set()
    t = 0.0

    while len(completed) < len(ps):
        ready = [
            i
            for i, p in enumerate(ps)
            if i not in completed and p.arrival <= t + EPS
        ]

        if not ready:
            next_arrival = min(
                p.arrival
                for i, p in enumerate(ps)
                if i not in completed
            )

            timeline.append(_segment("IDLE", t, next_arrival))
            t = next_arrival
            continue

        # Shortest burst first.
        # Exact SJF ties: earlier arrival, then lower PID.
        i = min(
            ready,
            key=lambda j: (
                ps[j].burst,
                ps[j].arrival,
                ps[j].pid,
            ),
        )

        p = ps[i]
        p.first_start = t

        end = t + p.burst
        timeline.append(_segment(i, t, end))

        t = end
        p.remaining = 0.0
        p.completion = t
        completed.add(i)

    return _build_result(ps, timeline)


# ---------------------------------------------------------------------
# Round Robin, q = 3
# ---------------------------------------------------------------------

def _rr_q3(
    data: Sequence[Tuple[float, float]],
) -> Dict[str, object]:
    ps = _make_processes(data)
    timeline: List[Dict[str, object]] = []

    quantum = 3.0

    arrival_order = sorted(
        range(len(ps)),
        key=lambda i: (ps[i].arrival, ps[i].pid),
    )

    ready: deque[int] = deque()

    next_arrival_index = 0
    t = 0.0

    while ready or next_arrival_index < len(ps):
        if not ready:
            next_pid = arrival_order[next_arrival_index]
            next_time = ps[next_pid].arrival

            if t < next_time - EPS:
                timeline.append(_segment("IDLE", t, next_time))
                t = next_time

            while (
                next_arrival_index < len(ps)
                and ps[arrival_order[next_arrival_index]].arrival
                <= t + EPS
            ):
                ready.append(arrival_order[next_arrival_index])
                next_arrival_index += 1

        i = ready.popleft()
        p = ps[i]

        if p.first_start is None:
            p.first_start = t

        run = min(quantum, p.remaining)
        end = t + run

        timeline.append(_segment(i, t, end))

        # Processes arriving during this time slice enter the queue
        # before the preempted process is appended again.
        while (
            next_arrival_index < len(ps)
            and ps[arrival_order[next_arrival_index]].arrival
            <= end + EPS
        ):
            ready.append(arrival_order[next_arrival_index])
            next_arrival_index += 1

        p.remaining -= run
        t = end

        if p.remaining > EPS:
            ready.append(i)
        else:
            p.remaining = 0.0
            p.completion = t

    return _build_result(ps, timeline)


# ---------------------------------------------------------------------
# CSA
# ---------------------------------------------------------------------

def _csa(
    data: Sequence[Tuple[float, float]],
) -> Dict[str, object]:
    ps = _make_processes(data)

    # L stores process IDs.
    L: List[int] = []

    # QUEUE stores {"pid": process_id, "n": jump_counter}.
    queue: List[Dict[str, int]] = []

    timeline: List[Dict[str, object]] = []
    state_table: List[Dict[str, object]] = []
    tie_events: List[Dict[str, object]] = []

    cnt = 0
    q_old = 0.0
    t = 0.0

    # -----------------------------
    # CSA helper functions
    # -----------------------------

    def add_arrived(now: float) -> List[int]:
        added: List[int] = []

        for p in ps:
            if (
                not p.admitted
                and p.remaining > EPS
                and p.arrival <= now + EPS
            ):
                p.admitted = True
                L.append(p.pid)
                added.append(p.pid)

        return added


    def select_process() -> int:
        """
        Paper's Select_Process(CNT) with deterministic tie handling.

        Odd CNT:
            minimum remaining time

        Even CNT:
            arrival time closest to
            (maximum arrival + minimum arrival) / 2

        Tie:
            oldest process = earliest arrival time

        Still tied:
            lower PID
        """

        if not L:
            raise RuntimeError("CSA Select_Process called with empty L")

        # Odd CNT: minimum remaining time
        if cnt % 2 == 1:
            minimum_remaining = min(
                ps[i].remaining for i in L
            )

            candidates = [
                i for i in L
                if abs(
                    ps[i].remaining - minimum_remaining
                ) <= EPS
            ]

            rule = "minimum remaining time"

        # Even CNT: arrival closest to midpoint
        else:
            arrivals = [ps[i].arrival for i in L]

            midpoint = (
                min(arrivals) + max(arrivals)
            ) / 2.0

            minimum_distance = min(
                abs(ps[i].arrival - midpoint)
                for i in L
            )

            candidates = [
                i for i in L
                if abs(
                    abs(ps[i].arrival - midpoint)
                    - minimum_distance
                ) <= EPS
            ]

            rule = "arrival closest to midpoint"

        if len(candidates) == 1:
            return candidates[0]

        # Tie breaker: oldest process.
        selected = min(
            candidates,
            key=lambda i: (
                ps[i].arrival,
                ps[i].pid,
            ),
        )

        tie_events.append({
            "time": t,
            "cnt": cnt,
            "rule": rule,
            "candidates": [
                f"P{i}" for i in candidates
            ],
            "selected": f"P{selected}",
            "tie_break": (
                "earliest arrival time, then lower PID"
            ),
        })

        return selected


    def active_ids_for_quantum() -> List[int]:
        ids: List[int] = []

        for i in L:
            if ps[i].remaining > EPS and i not in ids:
                ids.append(i)

        for entry in queue:
            i = entry["pid"]

            if ps[i].remaining > EPS and i not in ids:
                ids.append(i)

        return ids


    def calculate_quantum(
        selected: int,
    ) -> float:
        nonlocal q_old

        # Paper Step 4
        if len(L) == 1:
            if not queue:
                q_old = 1.0
                return 1.0

            return q_old if q_old >= 1.0 else 1.0

        active = active_ids_for_quantum()

        if not active:
            q_old = 1.0
            return 1.0

        remaining_values = [
            ps[i].remaining
            for i in active
        ]

        maximum_remaining = max(remaining_values)
        minimum_remaining = min(remaining_values)

        average_remaining = (
            sum(remaining_values)
            / len(remaining_values)
        )

        rtx = ps[selected].remaining

        terms = [
            rtx,
            (
                maximum_remaining
                - minimum_remaining
            ) / 2.0,
            average_remaining,
        ]

        # Paper says to remove zero values and calculate QNEW >= 1.
        # Clarification used here: retain fractional QNEW values.
        positive_terms = [
            x for x in terms if x > EPS
        ]

        raw_q = (
            min(positive_terms)
            if positive_terms
            else 1.0
        )

        q_new = max(1.0, raw_q)

        # No rounding added.
        q_old = q_new

        return q_new


    def calculate_jump() -> List[int]:
        returned: List[int] = []

        i = 0

        # Clarification used here: after removing an entry, restart at
        # the same index so a shifted entry is considered in this call.
        while i < len(queue):
            if queue[i]["n"] <= len(queue):

                # Decrement every queue counter.
                for entry in queue:
                    entry["n"] -= 1

                entry = queue.pop(i)
                pid = entry["pid"]

                # Unfinished process returns to L.
                if (
                    ps[pid].remaining > EPS
                    and pid not in L
                ):
                    L.append(pid)
                    returned.append(pid)

                # Do not increment i because the queue shifted.
            else:
                i += 1

        return returned


    def l_snapshot() -> List[str]:
        return [
            f"P{i}"
            for i in sorted(L)
        ]


    def queue_snapshot() -> List[Tuple[str, int]]:
        return [
            (
                f"P{entry['pid']}",
                entry["n"],
            )
            for entry in queue
        ]

    # -----------------------------
    # Initial state
    # -----------------------------

    add_arrived(t)

    state_table.append({
        "current_time": t,
        "cnt": cnt,
        "q_new": 0.0,
        "set_L": l_snapshot(),
        "queue": queue_snapshot(),
    })

    safety_counter = 0

    # -----------------------------
    # Main CSA loop
    # -----------------------------

    while True:
        safety_counter += 1

        if safety_counter > 100000:
            raise RuntimeError(
                "CSA exceeded safety iteration limit"
            )

        if (
            all(p.remaining <= EPS for p in ps)
            and not L
            and not queue
        ):
            break

        # Step 2:
        # add newly arrived processes
        add_arrived(t)

        # If L is empty, first try Calculate_Jump.
        if not L:
            calculate_jump()

        # Support workloads where no process arrives at t=0.
        if not L:
            future_arrivals = [
                p.arrival
                for p in ps
                if (
                    not p.admitted
                    and p.remaining > EPS
                )
            ]

            if future_arrivals:
                next_time = min(future_arrivals)

                if next_time > t + EPS:
                    timeline.append(
                        _segment(
                            "IDLE",
                            t,
                            next_time,
                        )
                    )

                    t = next_time

                add_arrived(t)

            elif queue:
                raise RuntimeError(
                    "CSA reached L empty while unfinished "
                    "processes remain only in QUEUE"
                )

        # Increment CNT before Select_Process,
        # following the paper.
        cnt += 1

        selected = select_process()

        q_new = calculate_quantum(selected)

        p = ps[selected]

        start = t

        if p.first_start is None:
            p.first_start = start

        run = min(
            q_new,
            p.remaining,
        )

        end = start + run

        timeline.append(
            _segment(
                selected,
                start,
                end,
            )
        )

        p.remaining -= run

        if p.remaining <= EPS:
            p.remaining = 0.0
            p.completion = end

        # Processes that arrive during this execution.
        just_arrived = [
            other.pid
            for other in ps
            if (
                not other.admitted
                and other.remaining > EPS
                and other.arrival > start + EPS
                and other.arrival <= end + EPS
            )
        ]

        # Step 5:
        # remove selected process from L.
        L.remove(selected)

        n_temp = (
            len(L)
            + len(just_arrived)
            + 1
        )

        n_x = (
            n_temp
            + len(queue)
        )

        queue.append({
            "pid": selected,
            "n": n_x,
        })

        # Step 6:
        calculate_jump()

        # Paper-style state at end time.
        # Include jobs that arrived during this execution,
        # because they are available at the end time.
        state_l = list(L)

        for pid in just_arrived:
            if pid not in state_l:
                state_l.append(pid)

        state_table.append({
            "current_time": end,
            "cnt": cnt,
            "q_new": q_new,
            "set_L": [
                f"P{i}"
                for i in sorted(state_l)
            ],
            "queue": queue_snapshot(),
        })

        t = end

    result = _build_result(
        ps,
        timeline,
    )

    result["state_table"] = state_table
    result["tie_events"] = tie_events
    result["implementation_assumptions"] = {
        "tie_break": "earliest arrival time, then lower PID",
        "fractional_qnew": "retained; no integer rounding",
        "calculate_jump_iteration": (
            "continue at the same queue index after removal"
        ),
        "rr_quantum": 3,
        "sjf_variant": "non-preemptive",
    }

    return result


# ---------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------

def evaluate_schedulers(
    processes: Sequence[Sequence[Number]],
) -> Dict[str, Dict[str, object]]:
    """
    Evaluate FCFS, non-preemptive SJF, RR(q=3), and CSA.

    Example:
        data = [
            [0, 2],   # P0
            [0, 3],   # P1
            [2, 1],   # P2
        ]

        results = evaluate_schedulers(data)

        print(results["csa"]["avg_tat"])
        print(results["csa"]["avg_response_time"])
        print(results["csa"]["execution_sequence"])
    """

    data = _validate_input(processes)

    return {
        "fcfs": _fcfs(data),
        "sjf": _sjf(data),
        "rr_q3": _rr_q3(data),
        "csa": _csa(data),
    }


# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    from pprint import pprint

    sample = [
        [0, 2],   # P0 - Arrival Time : 0, and Burst Time: 2
        [0, 3],   # P1 - Arrival Time : 0, and Burst Time: 3
        [2, 1],   # P2 - Arrival Time : 2, and Burst Time: 1
    ]

    results = evaluate_schedulers(sample)

    pprint(results)
