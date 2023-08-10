"""
Datastructures representing a genome as a collection of chromosomes. A number of
methods allow to manipulate genome regions, and retrieve the altered sequence.

Classes:
    - Chromosome
    - Genome
    - Position
    - Fragment
    - Track
"""


from __future__ import annotations
from typing import Dict, Iterator, Optional, Tuple
from dataclasses import dataclass, field
import copy
import pyBigWig
import pyfastx
import numpy as np


class Chromosome:
    """Representation of a chromosome as a collection of fragments.
    Each fragment represents a (0-based, right-open) region of the original 
    genome.
    """

    def __init__(self, name: str, length: int):
        self.name = name
        # list rather than np.array, due to better insert/append performance
        self.frags = [Fragment(self.name, 0, length)]

    def __len__(self):
        """Returns the total chromosome length."""
        return self.boundaries[-1]

    @property
    def boundaries(self):
        """Get array of fragment boundaries, from the start to the end of the 
        chromosome."""
        # Memorize whether fragments have changed to avoid recomputing the same
        # values.
        frags_hash = hash(tuple(self.frags))
        try:
            if self._frags_hash == frags_hash:
                changed = False
            else:
                changed = True
        # On first access, required attrs are generated
        except AttributeError:
            self._frags_hash = frags_hash
            changed = True
        if changed:
            self._bds = np.cumsum([0] + [len(frag) for frag in self.frags])
        return self._bds

    def get_frag_bounds(self, coord: int) -> Tuple[int, Tuple[int, int]]:
        """Returns the index and boundaries of the fragment in which input
        coordinate falls. Return format is (id, (start, end))."""
        bounds = self.boundaries
        if coord >= bounds[-1]:
            raise ValueError(f"Coordinate out of bounds: {self.name}:{coord}")
        frag_id = max(0, np.searchsorted(bounds, coord, side="right") - 1)
        return (frag_id, (bounds[frag_id], bounds[frag_id + 1]))

    def clean_frags(self):
        """Purge 0-length fragments."""
        self.frags = [frag for frag in self.frags if len(frag)]

    def insert(self, position: int, frag_ins: Fragment):
        """Updates fragments by inserting a sequence in the chromosome."""
        bounds = self.boundaries
        # Append after the end of chromosome
        if position == len(self):
            frag_id = len(bounds)
        else:
            frag_id, (frag_start, _) = self.get_frag_bounds(position)
        if position in bounds:
            # Insertion right between two fragments, add a fragment.
            self.frags.insert(frag_id, frag_ins)
        else:
            # Insertion inside a fragment, split it and add fragment in between.
            frag_l, frag_r = self.frags.pop(frag_id).split(
                position - frag_start
            )
            for frag in [frag_r, frag_ins, frag_l]:
                self.frags.insert(frag_id, frag)

    def invert(self, start: int, end: int):
        """Updates fragments by inverting a portion of the chromosome.
        The interval is 0-based and right open [start;end[."""
        s_frag_id, (s_frag_start, _) = self.get_frag_bounds(start)
        e_frag_id, (e_frag_start, _) = self.get_frag_bounds(end - 1)
        s_start_dist = start - s_frag_start
        e_start_dist = end - e_frag_start

        # Inversion inside a single frag.: Split it in 3 and invert middle.
        if s_frag_id == e_frag_id:
            inv_size = end - start
            frag_l, frag_mr = self.frags.pop(s_frag_id).split(s_start_dist)
            frag_m, frag_r = frag_mr.split(inv_size)
            frag_m.flip()
            for frag in [frag_r, frag_m, frag_l]:
                self.frags.insert(s_frag_id, frag)
        else:
            # Split fragment where inversion starts, we'll flip the right part.
            start_l, start_r = self.frags.pop(s_frag_id).split(s_start_dist)
            for frag in [start_r, start_l]:
                self.frags.insert(s_frag_id, frag)
            s_frag_id += 1
            e_frag_id += 1
            # Split fragment where inversion ends we'll flip the left part.
            end_l, end_r = self.frags.pop(e_frag_id).split(e_start_dist)
            for frag in [end_r, end_l]:
                self.frags.insert(e_frag_id, frag)
            e_frag_id += 1
            # If fragments are contained in the inversion, invert and flip them.
            for frag_id in range(s_frag_id, e_frag_id):
                self.frags[frag_id].flip()
            self.frags[s_frag_id:e_frag_id] = self.frags[
                e_frag_id - 1 : s_frag_id - 1 : -1
            ]

        self.clean_frags()

    def delete(self, start: int, end: int):
        """Updates fragments by deleting a portion of the chromosome.
        The interval is 0-based and right open [start;end[."""
        s_frag_id, (s_frag_start, _) = self.get_frag_bounds(start)
        e_frag_id, (_, e_frag_end) = self.get_frag_bounds(end - 1)
        del_size = end - start
        start_dist = start - s_frag_start
        end_dist = e_frag_end - end
        # Deletion contained in a single fragment: split it and trim right part
        if e_frag_id == s_frag_id:
            start_l, start_r = self.frags.pop(s_frag_id).split(start_dist)
            start_r.start += del_size
            for frag in [start_r, start_l]:
                self.frags.insert(s_frag_id, frag)
        # Deletion spans multiple fragments
        else:
            # Deletion starts in frag, end gets trimmed
            self.frags[s_frag_id].end = self.frags[s_frag_id].start + start_dist

            # Fragments contained in deletion disappear
            for frag_id in range(s_frag_id + 1, e_frag_id):
                curr_start = self.frags[frag_id].start
                if self.frags[frag_id].end < end:
                    self.frags[frag_id].end = curr_start
                if self.frags[frag_id].start < end:
                    self.frags[frag_id].start = curr_start

            from copy import copy

            ori_end = copy(self.frags[e_frag_id])
            # Deletion ends in frag, trim left side
            self.frags[e_frag_id].start = self.frags[e_frag_id].end - end_dist

    def get_seq(self, fasta: pyfastx.Fasta) -> Iterator[str]:
        """Retrieve the chromosome sequence, as a generator yielding
        the sequence by fragment."""
        self.clean_frags()
        for frag in self.frags:
            strand = "-" if frag.is_reverse else "+"
            # Note: fasta.fetch is 1-based...
            yield fasta.fetch(
                frag.chrom, (int(frag.start + 1), (frag.end)), strand=strand,
            )


class Genome:
    """Collection of chromosomes allowing complex SVs such as translocations."""

    def __init__(self, fasta: pyfastx.Fasta):
        self.fasta = fasta
        self.chroms = {}
        for name, seq in fasta:
            self.chroms[name] = Chromosome(name, len(seq))

    def __len__(self):
        return sum([len(chrom) for chrom in self.chroms.values()])

    @property
    def chromsizes(self) -> Dict[str, int]:
        chromsizes = {}
        for chrom in self.chroms.values():
            chromsizes[chrom.name] = len(chrom)
        return chromsizes

    def delete(self, chrom: str, start: int, end: int):
        """Delete a genomic segment."""
        self.chroms[chrom].delete(start, end)

    def insert(self, chrom: str, position: int, frag: Fragment):
        """Insert a new genomic segment."""
        self.chroms[chrom].insert(position, frag)

    def invert(self, chrom: str, start: int, end: int):
        """Invert (i.e. flip) a genomic segment."""
        self.chroms[chrom].invert(start, end)

    def translocate(
        self,
        target_chrom: str,
        target_pos: int,
        source_region: Fragment,
        invert: bool = False,
    ):
        """Move a genomic segment to another genomic position."""
        frag_size = source_region.end - source_region.start
        self.chroms[target_chrom].insert(target_pos, source_region)
        if invert:
            self.chroms[target_chrom].invert(target_pos, target_pos + frag_size)
        self.chroms[source_region.chrom].delete(
            source_region.start, source_region.end
        )

    def duplicate(
        self,
        target_chrom: str,
        target_pos: int,
        source_region: Fragment,
        invert: bool = False,
    ):
        """Copy a genomic segment to another genomic position."""
        frag_size = source_region.end - source_region.start
        self.chroms[target_chrom].insert(target_pos, source_region)
        if invert:
            self.chroms[target_chrom].invert(target_pos, target_pos + frag_size)

    def get_seq(self) -> Dict[str, Iterator[str]]:
        """Retrieve the genomic sequence of each chromosome. Each chromosome's
        sequence is returned as a generator of (lazily-retrieved) fragment sequences."""
        seqs = {}
        for chrom in self.chroms.values():
            seqs[chrom.name] = chrom.get_seq(self.fasta)
        return seqs


@dataclass(order=True)
class Position:
    """A single position in the genome, defined by a chromosome, genomic
    position and optionally a sign.

    Attributes
    ----------

    chrom:
        The chromosome name.
    coord:
        The 0-based coordinate on the chromosome.
    sign:
        Whether the position is on the 5' (-) or 3' (+) side.
    description:
        Short description of the position.
    """

    chrom: str
    coord: int
    #  None means unknown, True means 3'
    sign: Optional[bool] = field(default=None, compare=False)
    description: Optional[str] = field(default=None)

    def __repr__(self):
        sign_symbol = {None: "", True: "+", False: "-"}
        return f"{self.chrom}:{self.coord}:{sign_symbol[self.sign]}"

    def has_sign(self) -> bool:
        """Whether sign information is available."""
        return not self.sign is None


@dataclass
class Fragment:
    """A region representing a DNA sequence. Coordinates are 0-based and 
    left-open.

    Attributes
    ----------

    chrom:
        Chromosome on which the fragment is located.
    start:
        Coordinate where the fragment starts on the chromosome (smallest).
    end: 
        Coordinate where the fragment ends (largest).
    is_reverse:
        Whether the Fragment is reverse or not.
    description:
        Description of the type of fragment.
    """

    chrom: str
    start: int
    end: int
    is_reverse: bool = field(default=False)
    description: str = field(default=False)

    def __post_init__(self,):
        if self.end < self.start:
            raise ValueError("end cannot be smaller than start.")
        if self.start < 0:
            raise ValueError("Coordinates cannot be negative.")

    def __len__(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        sign = "-" if self.is_reverse else "+"
        return f"{self.chrom}:{self.start}-{self.end}:{sign}"

    def __hash__(self):
        return hash(str(self))

    def middle(self) -> int:
        return (self.start + self.end) // 2

    def intersect(self, other: Fragment) -> int:
        """Return the length of intersection with another fragment.
        If there is no intersection, returns 0."""
        same_chrom = self.chrom == other.chrom
        separate = (self.start > other.end) or (self.end < other.start)
        if same_chrom and not separate:
            overlap_start = max(self.start, other.start)
            overlap_end = min(self.end, other.end)
            return overlap_end - overlap_start
        return 0

    def split(self, rel_pos: int) -> Tuple[Fragment, Fragment]:
        """Split the fragment at a position relative from the start."""
        if rel_pos < 0:
            raise ValueError("Cannot split on negative coord")
        if rel_pos > (self.end - self.start):
            raise ValueError(
                f"Cannot split {self} at position {rel_pos}: Beyond fragment end."
            )
        left_frag, right_frag = copy.copy(self), copy.copy(self)
        left_frag.end = self.start + rel_pos
        right_frag.start = self.start + rel_pos
        return (left_frag, right_frag)

    def flip(self):
        """Change fragment's sign."""
        self.is_reverse = not self.is_reverse

    def merge(self, other: Fragment) -> Optional[Fragment]:
        """Merge two fragments with adjacent genomic positions. If fragments
        cannot be merged (e.g. because they are not adjacent), returns None."""
        compat_chroms = self.chrom == other.chrom
        compat_signs = self.is_reverse != other.is_reverse
        compat_coords = (self.start == other.end) or (other.start == self.end)
        if compat_chroms and compat_signs and compat_coords:
            start = min(self.start, other.start)
            end = min(self.end, other.end)
            frag = Fragment(self.chrom, start, end, self.is_reverse)
        else:
            frag = None
        return frag


class Track:
    """A representation of a genomic track on a genome."""

    def __init__(
        self, bigwig: pyBigWig.bigWigFile, circular: Optional[bool] = True
    ):
        self.bigwig = bigwig
        self.circular = circular
        self.values = {}
        for chrom in self.bigwig.chroms():
            self.values[chrom] = self.bigwig.values(
                chrom, 0, self.bigwig.chroms()[chrom]
            )
