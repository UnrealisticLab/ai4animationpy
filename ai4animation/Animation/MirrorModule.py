# Copyright (c) Meta Platforms, Inc. and affiliates.
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Animation.Module import Module
from ai4animation.Animation.Motion import Motion
from ai4animation.Math import Rotation, Tensor, Transform, Vector3


class MirrorModule(Module):
    def __init__(
        self,
        motion: Motion,
        axis,  # Vector3
        correction,  # Vector3
        overrides=None,  # Dictionary Name -> Vector3
    ) -> None:
        super().__init__(motion)

        self.MirrorAxis = axis

        self.Symmetry = self.DetectSymmetry(self.Motion.Hierarchy.BoneNames)

        self.Correction = Rotation.Euler(Vector3.Zero(len(motion.Hierarchy.BoneNames)))
        for i, sym_idx in enumerate(self.Symmetry):
            self.Correction[i : i + 1] = (
                Rotation.Euler(correction) if sym_idx != i else Rotation.Euler(0, 0, 0)
            )

        if overrides is not None:
            for k, v in overrides.items():
                self.Correction[self.Motion.Hierarchy.GetBoneIndex([k])] = (
                    Rotation.Euler(v)
                )

        self.NeedsCorrection: bool = not Tensor.All(
            self.Correction == Rotation.Euler(0, 0, 0)
        )

    def Initialize(self):
        pass

    def GetName(self):
        return "Mirror"

    def GetBoneTransformations(self, frame_indices, bone_indices):
        bone_indices = [self.Symmetry[x] for x in bone_indices]
        transforms = self.Motion.Frames[frame_indices][:, bone_indices]
        transforms = Transform.GetMirror(transforms, self.MirrorAxis)
        if self.NeedsCorrection:
            local_update = Transform.R(
                self.Correction[bone_indices],
            ).reshape(1, len(bone_indices), 4, 4)
            transforms = Transform.Multiply(transforms, local_update)
        return transforms

    def GUI(self, editor):
        if Module.Visualize[MirrorModule]:
            pass

    def Draw(self, editor):
        if Module.Visualize[MirrorModule]:
            names = editor.Actor.GetBoneNames()
            positions = self.Motion.GetBonePositions(
                editor.Timestamp,
                self.Motion.GetBoneIndices(names),
                True,
            ).reshape(-1, 3)
            AI4Animation.Draw.Skeleton(
                None, positions, editor.Actor, bones=names, size=2.0
            )

            # if self.Actor is None:
            #     self.Actor = editor.Actor.CreateCopy()
            # self.Actor.SetTransforms(
            #     self.GetBoneTransformations(
            #         self.Motion.GetFrameIndices(editor.Timestamp),
            #         self.Motion.GetBoneIndices(
            #             self.Actor.GetBoneNames(),
            #         ),
            #     )
            # )
            # self.Actor.SyncToScene()

    def DetectSymmetry(self, joint_names):
        def TryAssign(value: str, bone: int):
            if value in name_to_idx:
                symmetry[bone] = name_to_idx[value]
                return True
            else:
                return False

        name_to_idx = {name: i for i, name in enumerate(joint_names)}
        symmetry = list(range(len(joint_names)))
        for i, boneName in enumerate(joint_names):
            if boneName is None:
                continue
            if "_l_" in boneName:
                if TryAssign(boneName.replace("_l_", "_r_"), i):
                    continue

            if "_r_" in boneName:
                if TryAssign(boneName.replace("_r_", "_l_"), i):
                    continue

            if "_left_" in boneName:
                if TryAssign(boneName.replace("_left_", "_right_"), i):
                    continue

            if "_right_" in boneName:
                if TryAssign(boneName.replace("_right_", "_left_"), i):
                    continue

            if "Left" in boneName:
                if TryAssign(boneName.replace("Left", "Right"), i):
                    continue

            if "Right" in boneName:
                if TryAssign(boneName.replace("Right", "Left"), i):
                    continue

            if boneName[0] == "l" and boneName[1] == "_":
                symmName = "r_" + boneName[2:]
                if TryAssign(symmName, i):
                    continue

            if boneName[0] == "r" and boneName[1] == "_":
                symmName = "l_" + boneName[2:]
                if TryAssign(symmName, i):
                    continue

            symmetry[i] = i

        # for i, boneName in enumerate(joint_names):
        #    print(boneName, name_to_idx[boneName], joint_names[symmetry[i]])

        return symmetry
