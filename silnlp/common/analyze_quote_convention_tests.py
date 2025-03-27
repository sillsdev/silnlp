import unittest
from typing import Union

from .analyze_quote_convention import (
    QuoteConventionAnalysis,
    analyze_usfm_quote_convention,
)
from .environment import SIL_NLP_ENV


class TestNormalize(unittest.TestCase):

    def _test_usfm_file_convention(self, expected_convention_name: str, partial_usfm_file_path: str) -> None:
        identified_quote_convention: Union[QuoteConventionAnalysis, None] = analyze_usfm_quote_convention(
            SIL_NLP_ENV.pt_projects_dir / partial_usfm_file_path, print_summary=False
        )
        identified_quote_convention_name: Union[str, None] = None
        if identified_quote_convention is not None:
            identified_quote_convention_name = identified_quote_convention.get_best_quote_convention().get_name()
        self.assertEqual(expected_convention_name, identified_quote_convention_name)

    def test_minority_language_usfm_file_quote_convention_detection(self) -> None:
        self._test_usfm_file_convention("standard_arabic", "BFT_2024_01_31/01GENBFT.SFM")
        self._test_usfm_file_convention("western_european", "bcw_2024_02_21/41MATbcw.SFM")
        self._test_usfm_file_convention("standard_english", "BART_2023_05_29/NT41MAT_BART.SFM")
        self._test_usfm_file_convention("standard_english", "ksrBible_2023_11_07/01GENksrBible.SFM")
        self._test_usfm_file_convention(None, "GGSP_2023_12_14/01GENGGSP.SFM")
        self._test_usfm_file_convention("non-standard_arabic", "HEJ_2024_01_23/01GENHEJ.SFM")
        self._test_usfm_file_convention("typewriter_western_european_variant", "KIS_2024_01_10/01GENKIS.SFM")
        self._test_usfm_file_convention("standard_english", "KONDA_AI_2024_02_13/01GENKONDA_AI.SFM")
        self._test_usfm_file_convention("standard_english", "KW03a1_2024_02_15/01GENKW03a1.SFM")
        self._test_usfm_file_convention("standard_english", "lmp_2024_02_16/41MATlmp.SFM")
        self._test_usfm_file_convention("standard_english", "MGZ_2024_02_10/41MATMGZ.SFM")
        self._test_usfm_file_convention("standard_english", "RAJ_2024_07_11/41MATRAJ.SFM")
        self._test_usfm_file_convention("standard_english", "Sii_2024_02_14/01GENSii.SFM")
        self._test_usfm_file_convention("standard_english", "TDD_2023_10_24/41MATTDD.SFM")
        self._test_usfm_file_convention("typewriter_french", "SWO_2024_02_22/41MATSWO.SFM")
        self._test_usfm_file_convention("standard_english", "cja_WCK_2023_12_15/41MATcja_WCK.SFM")
        self._test_usfm_file_convention("standard_french", "NTB_2024_07_17/41MATNTB.SFM")
        self._test_usfm_file_convention("standard_english", "LNT/41MATLMPNT03.SFM")
        self._test_usfm_file_convention("standard_english", "nhu_2024_02_16/41MATnhu.SFM")
        self._test_usfm_file_convention("standard_english", "NBT_2024_02_16/41MATNBT.SFM")
        self._test_usfm_file_convention("french_variant", "OMI2_2024_09_22/41MATOMI2.SFM")
        self._test_usfm_file_convention("standard_french", "buu_Nita/41MATBudNita8.SFM")
        self._test_usfm_file_convention("standard_english", "KakHaa/41MATBoranaNT.SFM")
        self._test_usfm_file_convention("typewriter_english", "KOT_2025_02_07/01GENKOT.SFM")
        self._test_usfm_file_convention("standard_french", "BTU_2024_03_12/MAT.BTU")
        self._test_usfm_file_convention("standard_english", "POV2014_2024_06_25/01GENPOV2014.SFM")
        self._test_usfm_file_convention("standard_english", "BTM_2024_07_03/41MATBTM.SFM")
        self._test_usfm_file_convention("standard_english", "ETU_2023_11_13/41MATETU.SFM")
        self._test_usfm_file_convention("western_european", "KBY_2023_11_09/41MATKBY.SFM")
        self._test_usfm_file_convention("hybrid_typewriter_western_european", "TTQ_2025_02_03/01GENTTQ.SFM")
        self._test_usfm_file_convention("hybrid_typewriter_english", "KRX_2024_03_04/01GENKRX.SFM")
        self._test_usfm_file_convention("typewriter_french", "cWol_2024_08_07/41MATcWol.SFM")
        self._test_usfm_file_convention("western_european", "KBYc_2025_03_11/41MATKBYc.SFM")

    def test_major_language_usfm_file_quote_convention_detection(self) -> None:
        self._test_usfm_file_convention("western_european", "DHH94EE/01GENDHHDE.SFM")
        self._test_usfm_file_convention("western_european", "RVR95/01GENRVEES95.SFM")
        self._test_usfm_file_convention("western_european", "LBLA/01GENLBLA.SFM")
        self._test_usfm_file_convention("western_european", "NBLA/01GENNBL.SFM")
        self._test_usfm_file_convention("british_inspired_western_european", "LBS21/01GENLBS21.SFM")
        self._test_usfm_file_convention(None, "SGNEG79/01GENSGNEG79.SFM")
        self._test_usfm_file_convention(None, "fraLSG/01GENfraLSG.usfm")
        self._test_usfm_file_convention("western_european", "FCR19DBL/01GENFCR19DBL.SFM")
        self._test_usfm_file_convention("western_european", "TOB/01GENTOB10.SFM")
        self._test_usfm_file_convention("standard_english", "NAA/01GENNA17.SFM")
        self._test_usfm_file_convention("standard_english", "NVI-P/01GENpor.SFM")
        self._test_usfm_file_convention("typewriter_french", "Ret4/01GENRet4.SFM")
        self._test_usfm_file_convention("typewriter_french", "TBT1_2024_01_11/01GENTBT1.SFM")
        self._test_usfm_file_convention("standard_english", "VFL/01GEN_D_POR.SFM")
        self._test_usfm_file_convention("british_english", "NIV11UK/01GENukNIV11.SFM")
        self._test_usfm_file_convention("standard_swedish", "NUB/01GENsweNUB15.SFM")
        self._test_usfm_file_convention("non-standard_arabic", "ONAV/01GENarONAV12.SFM")
        self._test_usfm_file_convention("western_european", "TGVD/01GENTGV03.SFM")
        self._test_usfm_file_convention("hybrid_british_typewriter_western_european", "TPV/01GENTPV.SFM")
        self._test_usfm_file_convention("standard_russian", "CARS/01GENruCARS13.SFM")
        self._test_usfm_file_convention("standard_russian", "NRT23/MATrusNRT23.SFM")
        self._test_usfm_file_convention("standard_hungarian", "BPH/01GENdanBPH15.SFM")
        self._test_usfm_file_convention("standard_german", "Bibelen2020/01GENDCT.SFM")
        self._test_usfm_file_convention("standard_german", "DBSV/01GENDBSV.SFM")
        self._test_usfm_file_convention("standard_german", "DO92apo/01GENDBDC.SFM")
        self._test_usfm_file_convention("standard_hungarian", "polsz/41MATpolsz.usfm")
        self._test_usfm_file_convention("standard_finnish", "Raamattu/01GENFIN92.SFM")


if __name__ == "__main__":
    unittest.main()
