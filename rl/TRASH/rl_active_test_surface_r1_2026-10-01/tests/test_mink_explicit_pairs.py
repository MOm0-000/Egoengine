"""Requested runtime pairs must survive MINK's ordinary adjacency filtering."""

import mink
import mujoco


def test_explicit_adjacent_pair_is_opt_in_and_does_not_add_unrequested_pairs():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom name="palm" size=".02" contype="0" conaffinity="0"/>
        <body pos=".045 0 0"><joint type="hinge"/>
          <geom name="root" size=".02" contype="0" conaffinity="0"/>
        </body>
        <body pos="0 .06 0"><joint type="hinge"/><geom name="other" size=".02"/></body>
      </body></worldbody><contact><pair geom1="palm" geom2="root"/></contact></mujoco>''')
    groups = [(["palm"], ["root", "other"])]
    assert mink.CollisionAvoidanceLimit(model, groups).geom_id_pairs == []
    limit = mink.CollisionAvoidanceLimit(model, groups, include_explicit_pairs=True)
    assert set(limit.geom_id_pairs) == {(model.geom("palm").id, model.geom("root").id)}
    assert not mink.CollisionAvoidanceLimit(model, [], include_explicit_pairs=True).geom_id_pairs
    configuration = mink.Configuration(model)
    assert limit.compute_qp_inequalities(configuration, .01).G.shape[1] == model.nv
