"""Score-level dual-model ensemble for the assembly pipeline mode.

The ensemble exposes a :class:`NeuralCategModel`-compatible interface
(``classifyDataset(...)``, ``catList``, ``save(...)``) so that the existing
:func:`evaluate.evaluate_predictions` and :func:`legacy_reuse.evaluateModel`
flow consume it without any code changes.

For each class, the assembly returns the score from the member assigned to
that class by :class:`ClassToModelMap`. Per-class threshold filtering is
applied later by :func:`evaluate.filter_and_normalize` via
``customThresholds`` — i.e. the assembly does not threshold itself.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassToModelMap:
    """Per-class assembly member selection.

    :param assignments: ``{cat_id -> member_index}`` (0 = primary).
    :param member_labels: Human-readable labels indexed by member index.
    """

    assignments: Mapping[str, int]
    member_labels: tuple[str, ...]


class AssemblyModel:
    """Two-member score-level ensemble with per-class model selection.

    Mirrors the public surface of ``NeuralCategModel`` used by the eval
    pipeline:

    - :pyattr:`catList` — the shared category list.
    - :pymeth:`classifyDataset` — combined per-class scores per document.
    - :pymeth:`save` — persist member bundles + manifest.

    Threshold filtering and IPTC normalization happen downstream in
    :func:`evaluate.filter_and_normalize`. ``classifyDataset`` returns ALL
    per-class scores (filtered to the assigned member only) so the standard
    ``customThresholds`` mechanism can apply per-class cutoffs.
    """

    def __init__(
        self,
        *,
        members: Sequence[Any],
        cat_list: Sequence[str],
        class_to_model: ClassToModelMap,
    ) -> None:
        if len(members) < 1:
            raise ValueError('AssemblyModel requires at least one member')
        if len(class_to_model.member_labels) != len(members):
            raise ValueError(
                f'member_labels has {len(class_to_model.member_labels)} entries '
                f'but {len(members)} members were provided'
            )
        unknown_cats = set(class_to_model.assignments) - set(cat_list)
        if unknown_cats:
            raise ValueError(
                f'class_to_model.assignments has {len(unknown_cats)} categories '
                f'not in cat_list (sample={sorted(unknown_cats)[:5]})'
            )
        bad_idx = {
            cid: idx for cid, idx in class_to_model.assignments.items()
            if idx < 0 or idx >= len(members)
        }
        if bad_idx:
            raise ValueError(
                f'class_to_model.assignments contains out-of-range member indices: '
                f'{dict(list(bad_idx.items())[:5])}'
            )

        self._members = tuple(members)
        self.catList = list(cat_list)
        self._class_to_model = class_to_model
        self._owned: tuple[set[str], ...] = tuple(
            {cid for cid, idx in class_to_model.assignments.items() if idx == m_idx}
            for m_idx in range(len(members))
        )
        self._device = getattr(members[0], '_device', None)

    @property
    def class_to_model(self) -> ClassToModelMap:
        """Expose the per-class assignment map."""
        return self._class_to_model

    @property
    def members(self) -> tuple[Any, ...]:
        """Expose the underlying member models."""
        return self._members

    def classifyDataset(
        self,
        evalData: Any,
        thr: float = -9999.0,
        returnScores: bool = True,
        sigmoidScores: bool = True,
    ) -> list[list[tuple[str, float]]]:
        """
        Return per-document scored labels combined per class from members.

        For class ``c`` assigned to member ``m``, the returned ``(c, score)``
        pair is member ``m``'s sigmoid score for ``c``. Classes outside
        :pyattr:`catList` are dropped.

        :param evalData: Dataset to score (each member receives the same
            evaluation dataset since the per-member feature alignment is
            handled at dataset-construction time).
        :param thr: Forwarded to each member. Use the pipeline's default
            ``threshold_predict`` (``-9999``) so all scores survive — the
            per-class thresholding happens later in
            :func:`evaluate.filter_and_normalize`.
        :param returnScores: Kept for signature parity; the assembly always
            returns scored labels.
        :param sigmoidScores: Forwarded to each member.
        :return: ``list[list[(cat_id, score)]]`` aligned with the docs in
            ``evalData``.
        """
        per_member = [
            m.classifyDataset(
                evalData, thr=thr, returnScores=True, sigmoidScores=sigmoidScores,
            )
            for m in self._members
        ]
        n_docs = len(per_member[0])
        for idx, preds in enumerate(per_member):
            if len(preds) != n_docs:
                raise ValueError(
                    f'AssemblyModel: member={idx} produced {len(preds)} docs '
                    f'vs primary {n_docs}'
                )

        out: list[list[tuple[str, float]]] = []
        for doc_idx in range(n_docs):
            combined: list[tuple[str, float]] = []
            for m_idx, doc_preds in enumerate(per_member):
                owned = self._owned[m_idx]
                if not owned:
                    continue
                for cat_id, score in doc_preds[doc_idx]:
                    if cat_id in owned:
                        combined.append((str(cat_id), float(score)))
            combined.sort(key=lambda kv: kv[1], reverse=True)
            out.append(combined)
        return out

    def save(self, path: str) -> None:
        """
        Persist members side-by-side and a manifest pointing at them.

        Each member is saved at ``{path}.member_{i}_{label}`` and a JSON
        manifest is written at ``{path}.assembly_manifest.json`` next to
        the requested ``path``. The ``path`` argument itself is treated
        as the manifest's primary anchor — the manifest's path replicates
        the conventional ``model.nn.bin`` naming used by the rest of the
        pipeline.
        """
        anchor = Path(path)
        anchor.parent.mkdir(parents=True, exist_ok=True)

        member_paths: list[str] = []
        labels = self._class_to_model.member_labels
        for idx, member in enumerate(self._members):
            member_path = anchor.with_name(f'{anchor.name}.member_{idx}_{labels[idx]}')
            member.save(str(member_path))
            member_paths.append(str(member_path))

        manifest_path = anchor.with_name(f'{anchor.name}.assembly_manifest.json')
        manifest = {
            'schema_version': '1',
            'anchor_path': str(anchor),
            'member_labels': list(labels),
            'member_paths': member_paths,
            'cat_list': list(self.catList),
            'assignments': {str(k): int(v) for k, v in self._class_to_model.assignments.items()},
        }
        with manifest_path.open('w', encoding='utf-8') as out:
            json.dump(manifest, out, ensure_ascii=False, indent=2, sort_keys=True)
        LOGGER.info(
            f'Assembly: saved manifest path={manifest_path} '
            f'members={len(self._members)} n_classes={len(self.catList)}'
        )
