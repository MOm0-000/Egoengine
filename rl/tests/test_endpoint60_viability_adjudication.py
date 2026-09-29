from scripts.summarize_taco_pour_endpoint60_viability_v1 import classify


def test_carryover_has_highest_precedence():
    assert classify(carryover=True, A=True, B=True, P=True) == (
        "C1", "POLICY_CARRYOVER_SOLVES_NEXT_WINDOW"
    )


def test_only_promoted_arrival_feasible_is_c3():
    assert classify(carryover=False, A=False, B=False, P=True) == (
        "C3", "ARRIVAL_STATE_VIABILITY_BOTTLENECK"
    )


def test_any_legacy_arrival_feasible_is_c2_regardless_of_p():
    for A, B, P in ((True, False, False), (False, True, True), (True, True, True)):
        assert classify(carryover=False, A=A, B=B, P=P) == (
            "C2", "ONE_STEP_VIABLE_ACTION_EXISTS"
        )


def test_no_known_arrival_feasible_is_finite_negative_c4():
    assert classify(carryover=False, A=False, B=False, P=False) == (
        "C4", "NO_ONE_STEP_FEASIBLE_ACTION_FOUND_ON_KNOWN_ENDPOINT60_STATES"
    )
