"""Frozen hand-side choices for the single-hand TACO-16 physics evaluation.

These choices are data provenance, not a convenience default.  A physical
grasp score is meaningful only for the hand selected here and must be reported
with that side.  ``measure_ruler_cup`` is bimanual in GT; it remains a declared
right-hand *single-hand* proxy until a bimanual evaluator is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TacoEvalEpisode:
    episode: str
    action_dir: str
    side: str
    hand_selection_note: str

    @property
    def reference_stem(self) -> str:
        """Side-qualified references avoid overwriting legacy right-hand files."""
        return "reference_robot_trajectory_v2" if self.side == "right" else (
            f"reference_robot_trajectory_v2_{self.side}"
        )

    @property
    def human_reference_name(self) -> str:
        return "human_reference_v2.npz" if self.side == "right" else (
            f"human_reference_v2_{self.side}.npz"
        )


_RIGHT = "GT-contacting hand selected as right"

TACO_EVAL16: tuple[TacoEvalEpisode, ...] = (
    TacoEvalEpisode("taco_brush_brush_bowl_20230927_027", "taco_brush_brush_bowl_20230927_027", "right", _RIGHT),
    TacoEvalEpisode("taco_brush_roller_plate_20230928_043", "taco_brush_roller_plate_20230928_043", "right", _RIGHT),
    TacoEvalEpisode("taco_cut_spatula_plate_20230917_020", "taco_cut_spatula_plate_20230917_020", "right", _RIGHT),
    TacoEvalEpisode("taco_dust_brush_box_20230927_030", "taco_dust_brush_box_20230927_030", "right", _RIGHT),
    TacoEvalEpisode("taco_empty_kettle_bowl_20231019_002", "empty_kettle", "left", "GT-contacting hand selected as left (right has zero GT contact frames)"),
    TacoEvalEpisode("taco_hit_hammer_toy_20231102_051", "taco_hit_hammer_toy_20231102_051", "right", _RIGHT),
    TacoEvalEpisode("taco_measure_ruler_cup_20231104_101", "taco_measure_ruler_cup_20231104_101", "right", "GT is bimanual (left/right contacts 63/62); declared right-hand single-hand proxy, not a bimanual score"),
    TacoEvalEpisode("taco_pour_kettle_cup_20230917_036", "taco_pour_kettle_cup_20230917_036", "right", _RIGHT),
    TacoEvalEpisode("taco_put_in_spoon_bowl_20231104_179", "taco_put_in_spoon_bowl_20231104_179", "right", _RIGHT),
    TacoEvalEpisode("taco_put_out_spatula_plate_20231015_118", "taco_put_out_spatula_plate_20231015_118", "right", _RIGHT),
    TacoEvalEpisode("taco_scrape_knife_pan_20231031_126", "taco_scrape_knife_pan_20231031_126", "right", _RIGHT),
    TacoEvalEpisode("taco_screw_screwdriver_box_20231102_009", "taco_screw_screwdriver_box_20231102_009", "right", _RIGHT),
    TacoEvalEpisode("taco_skim_spatula_plate_20230926_004", "taco_skim_spatula_plate_20230926_004", "right", _RIGHT),
    TacoEvalEpisode("taco_smear_eraser_box_20231103_071", "taco_smear_eraser_box_20231103_071", "right", _RIGHT),
    TacoEvalEpisode("taco_stir_fry_spatula_pan_20231013_006", "taco_stir_fry_spatula_pan_20231013_006", "right", _RIGHT),
    TacoEvalEpisode("taco_stir_spoon_bowl_20231104_184", "taco_stir_spoon_bowl_20231104_184", "right", _RIGHT),
)
