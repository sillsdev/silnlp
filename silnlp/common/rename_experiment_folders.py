"""Rename existing experiment folders in the bucket to the names the onboarding tools generate.

Usage: python -m silnlp.common.rename_experiment_folders
"""

import argparse
import subprocess
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Rename:
    old: str
    new: str


class ExperimentRenames:
    def all(self) -> List[Rename]:
        countries = [
            ("CookIslands", "Cook_Islands"),
            ("EquatorialGuinea", "Equatorial_Guinea"),
            ("NorthMacedonia", "North_Macedonia"),
            ("SouthSudan", "South_Sudan"),
            ("DRC", "DR_Congo"),
            ("BurkinaFaso", "Burkina_Faso"),
            ("Solomons", "Solomon_Islands"),
            ("Surinam", "Suriname"),
            ("Vietnam", "Viet_Nam"),
            ("IvoryCoast", "Côte_d'Ivoire"),
            ("CAR", "Central_African_Republic"),
        ]
        languages = [
            ("Cambodia", "Cham_Western", "Western_Cham"),
            ("India", "North_Western_Kolami", "Northwestern_Kolami"),
            ("India", "DigaroMishmi", "Digaro_Mishmi"),
            ("India", "IduMishmi", "Idu_Mishmi"),
            ("India", "KhariaThar", "Kharia_Thar"),
            ("India", "BrajBhasha", "Braj_Bhasha"),
            ("India", "KondaDora", "Konda_Dora"),
            ("India", "Muria_Far_Western", "Far_Western_Muria"),
            ("Mexico", "Purepecha_Western_Highland", "Western_Highland_Purepecha"),
            ("Mexico", "SoutheasternPueblaNahuatl", "Southeastern_Puebla_Nahuatl"),
            ("Nepal", "WesternTamang", "Western_Tamang"),
            ("Nigeria", "NyaHuba", "Nya_Huba"),
            ("Algeria", "AlgerianArabic", "Algerian_Arabic"),
            ("Egypt", "EgyptianArabic", "Egyptian_Arabic"),
            ("Saudi_Arabia", "Arabic_Standard", "Standard_Arabic"),
            ("Tunisia", "TunisianArabic", "Tunisian_Arabic"),
            ("Laos", "NorthernKatang", "Northern_Katang"),
            ("Myanmar", "MuakSaaak", "Muak_Sa_aak"),
            ("Vanuatu", "NorthAmbrym", "North_Ambrym"),
            ("Vanuatu", "NorthTanna", "North_Tanna"),
            ("PNG", "AropLokep", "Arop_Lokep"),
            ("PNG", "BenaBena", "Benabena"),
            ("PNG", "Kuni-Boazi", "Kuni_Boazi"),
            ("PNG", "Doromu-Koki", "Doromu_Koki"),
            ("PNG", "PinaiHagahai", "Pinai_Hagahai"),
            ("Indonesia", "SekoPadang", "Seko_Padang"),
            ("Pakistan", "SindhiBhil", "Sindhi_Bhil"),
            ("Guinea", "BagaSitemu", "Baga_Sitemu"),
            ("Congo", "TekeTyee", "Teke_Tyee"),
            ("Congo", "TekeEboo", "Teke_Eboo"),
            ("Liberia", "EasternKrahn", "Eastern_Krahn"),
            ("Niger", "KanuriManga", "Kanuri_Manga"),
            ("Uganda", "SouthernMadi", "Southern_Ma’di"),
            ("Mexico", "ChatinoZenzontepec", "Chatino_Zenzontepec"),
            ("Thailand", "NorthernThai", "Northern_Thai"),
            ("Haiti", "HaitianCreole", "Haitian_Creole"),
            ("Peru", "Yanesha", "Yanesha’"),
        ]
        # Countries first, so a language folder is never moved under a country that is about to be renamed.
        return [Rename(old, new) for old, new in countries] + [
            Rename(f"{country}/{old}", f"{country}/{new}") for country, old, new in languages
        ]


class ExperimentsRemote:
    def __init__(self, root: str):
        self._root = root.rstrip("/")

    def exists(self, folder: str) -> bool:
        # rclone succeeds for a missing folder, and object storage has no empty folders, so non-empty means present.
        listing = subprocess.run(
            ["rclone", "lsf", "--max-depth", "1", f"{self._root}/{folder}"], capture_output=True, text=True
        )
        return listing.returncode == 0 and bool(listing.stdout.strip())

    def move(self, old: str, new: str) -> bool:
        # --ignore-existing so that merging into a folder that already exists never overwrites its files.
        command = [
            "rclone",
            "move",
            "--ignore-existing",
            "--delete-empty-src-dirs",
            f"{self._root}/{old}",
            f"{self._root}/{new}",
        ]
        return subprocess.run(command).returncode == 0


class Prompt:
    def answer(self, question: str) -> str:
        try:
            return input(f"{question}: ").strip().lower()
        except EOFError:
            return "q"


class RenameSession:
    def __init__(self, remote: ExperimentsRemote, prompt: Prompt):
        self._remote = remote
        self._prompt = prompt

    def run(self, renames: List[Rename]) -> int:
        todo = self._show_plan(renames)
        if not todo:
            return 0
        reply = self._prompt.answer(
            f"Rename these {len(todo)} folders one after another? y = all, s = ask for each one, anything else = none"
        )
        if reply in ("y", "yes"):
            return self._move_all(todo)
        if reply == "s":
            return self._move_each(todo)
        print("Nothing renamed.")
        return 0

    def _show_plan(self, renames: List[Rename]) -> List[Rename]:
        todo = []
        for rename in renames:
            if not self._remote.exists(rename.old):
                print(f"  not found, skipping: {rename.old}")
            elif self._remote.exists(rename.new):
                print(f"  MERGE (existing files are kept): {rename.old} -> {rename.new}")
                todo.append(rename)
            else:
                print(f"  {rename.old} -> {rename.new}")
                todo.append(rename)
        return todo

    def _move_all(self, todo: List[Rename]) -> int:
        failed = []
        for number, rename in enumerate(todo, 1):
            print(f"[{number}/{len(todo)}] {rename.old} -> {rename.new}", flush=True)
            if not self._move(rename):
                failed.append(rename)
        done = len(todo) - len(failed)
        print(f"Renamed {done} of {len(todo)} folders.")
        for rename in failed:
            print(f"  failed, run again to retry: {rename.old}")
        return done

    def _move_each(self, todo: List[Rename]) -> int:
        done = 0
        for rename in todo:
            reply = self._prompt.answer(f"Rename {rename.old} -> {rename.new}? y/N/q")
            if reply == "q":
                break
            if reply in ("y", "yes") and self._move(rename):
                done += 1
        print(f"Renamed {done} folder(s).")
        return done

    def _move(self, rename: Rename) -> bool:
        if not self._remote.move(rename.old, rename.new):
            print(f"Failed to move {rename.old}.")
            return False
        if self._remote.exists(rename.old):
            print(f"{rename.old} still holds files that were not moved because {rename.new} has the same names.")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="miniosilnlp:nlp-research/MT/experiments", help="rclone path of the experiments folder"
    )
    args = parser.parse_args()
    RenameSession(ExperimentsRemote(args.root), Prompt()).run(ExperimentRenames().all())


if __name__ == "__main__":
    main()
