import argparse
import csv
import json
from pathlib import Path

from ..common.environment import SIL_NLP_ENV
NLLB_LANG_CODES = {'ace': 'ace_Latn', 'acm': 'acm_Arab', 'acq': 'acq_Arab', 'aeb': 'aeb_Arab', 'afr': 'afr_Latn', 'ajp': 'ajp_Arab', 'aka': 'aka_Latn', 'amh': 'amh_Ethi', 'apc': 'apc_Arab', 'arb': 'arb_Arab', 'ars': 'ars_Arab', 'ary': 'ary_Arab', 'arz': 'arz_Arab', 'asm': 'asm_Beng', 'ast': 'ast_Latn', 'awa': 'awa_Deva', 'ayr': 'ayr_Latn', 'azb': 'azb_Arab', 'azj': 'azj_Latn', 'bak': 'bak_Cyrl', 'bam': 'bam_Latn', 'ban': 'ban_Latn', 'bel': 'bel_Cyrl', 'bem': 'bem_Latn', 'ben': 'ben_Beng', 'bho': 'bho_Deva', 'bjn': 'bjn_Latn', 'bod': 'bod_Tibt', 'bos': 'bos_Latn', 'bug': 'bug_Latn', 'bul': 'bul_Cyrl', 'cat': 'cat_Latn', 'ceb': 'ceb_Latn', 'ces': 'ces_Latn', 'cjk': 'cjk_Latn', 'ckb': 'ckb_Arab', 'crh': 'crh_Latn', 'cym': 'cym_Latn', 'dan': 'dan_Latn', 'deu': 'deu_Latn', 'dik': 'dik_Latn', 'dyu': 'dyu_Latn', 'dzo': 'dzo_Tibt', 'ell': 'ell_Grek', 'eng': 'eng_Latn', 'epo': 'epo_Latn', 'est': 'est_Latn', 'eus': 'eus_Latn', 'ewe': 'ewe_Latn', 'fao': 'fao_Latn', 'pes': 'pes_Arab', 'fij': 'fij_Latn', 'fin': 'fin_Latn', 'fon': 'fon_Latn', 'fra': 'fra_Latn', 'fur': 'fur_Latn', 'fuv': 'fuv_Latn', 'gla': 'gla_Latn', 'gle': 'gle_Latn', 'glg': 'glg_Latn', 'grn': 'grn_Latn', 'guj': 'guj_Gujr', 'hat': 'hat_Latn', 'hau': 'hau_Latn', 'heb': 'heb_Hebr', 'hin': 'hin_Deva', 'hne': 'hne_Deva', 'hrv': 'hrv_Latn', 'hun': 'hun_Latn', 'hye': 'hye_Armn', 'ibo': 'ibo_Latn', 'ilo': 'ilo_Latn', 'ind': 'ind_Latn', 'isl': 'isl_Latn', 'ita': 'ita_Latn', 'jav': 'jav_Latn', 'jpn': 'jpn_Jpan', 'kab': 'kab_Latn', 'kac': 'kac_Latn', 'kam': 'kam_Latn', 'kan': 'kan_Knda', 'kas': 'kas_Deva', 'kat': 'kat_Geor', 'knc': 'knc_Latn', 'kaz': 'kaz_Cyrl', 'kbp': 'kbp_Latn', 'kea': 'kea_Latn', 'khm': 'khm_Khmr', 'kik': 'kik_Latn', 'kin': 'kin_Latn', 'kir': 'kir_Cyrl', 'kmb': 'kmb_Latn', 'kon': 'kon_Latn', 'kor': 'kor_Hang', 'kmr': 'kmr_Latn', 'lao': 'lao_Laoo', 'lvs': 'lvs_Latn', 'lij': 'lij_Latn', 'lim': 'lim_Latn', 'lin': 'lin_Latn', 'lit': 'lit_Latn', 'lmo': 'lmo_Latn', 'ltg': 'ltg_Latn', 'ltz': 'ltz_Latn', 'lua': 'lua_Latn', 'lug': 'lug_Latn', 'luo': 'luo_Latn', 'lus': 'lus_Latn', 'mag': 'mag_Deva', 'mai': 'mai_Deva', 'mal': 'mal_Mlym', 'mar': 'mar_Deva', 'min': 'min_Latn', 'mkd': 'mkd_Cyrl', 'plt': 'plt_Latn', 'mlt': 'mlt_Latn', 'mni': 'mni_Beng', 'khk': 'khk_Cyrl', 'mos': 'mos_Latn', 'mri': 'mri_Latn', 'zsm': 'zsm_Latn', 'mya': 'mya_Mymr', 'nld': 'nld_Latn', 'nno': 'nno_Latn', 'nob': 'nob_Latn', 'npi': 'npi_Deva', 'nso': 'nso_Latn', 'nus': 'nus_Latn', 'nya': 'nya_Latn', 'oci': 'oci_Latn', 'gaz': 'gaz_Latn', 'ory': 'ory_Orya', 'pag': 'pag_Latn', 'pan': 'pan_Guru', 'pap': 'pap_Latn', 'pol': 'pol_Latn', 'por': 'por_Latn', 'prs': 'prs_Arab', 'pbt': 'pbt_Arab', 'quy': 'quy_Latn', 'ron': 'ron_Latn', 'run': 'run_Latn', 'rus': 'rus_Cyrl', 'sag': 'sag_Latn', 'san': 'san_Deva', 'sat': 'sat_Beng', 'scn': 'scn_Latn', 'shn': 'shn_Mymr', 'sin': 'sin_Sinh', 'slk': 'slk_Latn', 'slv': 'slv_Latn', 'smo': 'smo_Latn', 'sna': 'sna_Latn', 'snd': 'snd_Arab', 'som': 'som_Latn', 'sot': 'sot_Latn', 'spa': 'spa_Latn', 'als': 'als_Latn', 'srd': 'srd_Latn', 'srp': 'srp_Cyrl', 'ssw': 'ssw_Latn', 'sun': 'sun_Latn', 'swe': 'swe_Latn', 'swh': 'swh_Latn', 'szl': 'szl_Latn', 'tam': 'tam_Taml', 'tat': 'tat_Cyrl', 'tel': 'tel_Telu', 'tgk': 'tgk_Cyrl', 'tgl': 'tgl_Latn', 'tha': 'tha_Thai', 'tir': 'tir_Ethi', 'taq': 'taq_Tfng', 'tpi': 'tpi_Latn', 'tsn': 'tsn_Latn', 'tso': 'tso_Latn', 'tuk': 'tuk_Latn', 'tum': 'tum_Latn', 'tur': 'tur_Latn', 'twi': 'twi_Latn', 'tzm': 'tzm_Tfng', 'uig': 'uig_Arab', 'ukr': 'ukr_Cyrl', 'umb': 'umb_Latn', 'urd': 'urd_Arab', 'uzn': 'uzn_Latn', 'vec': 'vec_Latn', 'vie': 'vie_Latn', 'war': 'war_Latn', 'wol': 'wol_Latn', 'xho': 'xho_Latn', 'ydd': 'ydd_Hebr', 'yor': 'yor_Latn', 'yue': 'yue_Hant', 'zho': 'zho_Hant', 'zul': 'zul_Latn'}
NLLB_ISOS = NLLB_LANG_CODES.keys()


def load_language_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    # Restructure the data for faster lookups
    language_data = {}
    country_data = {}
    family_data = {}

    for lang in raw_data:
        iso = lang["isoCode"]
        country = lang["langCountry"]
        family = lang["languageFamily"]

        language_data[iso] = {
            "Name": lang["language"],
            "Country": country,
            "Family": family,
        }

        country_data.setdefault(country, []).append(iso)
        family_data.setdefault(family, []).append(iso)

    return language_data, country_data, family_data
   

def process_iso_codes(iso_codes, language_data, country_data, family_data, nllb_isos):
    iso_set = set(iso_codes)

    for iso in iso_codes:
        if iso in language_data:

            lang_info = language_data[iso]
            print(
                f"{iso}: {lang_info['Name']}, {lang_info['Country']}, {lang_info['Family']}"
            )

            # Add iso codes from the same country
            iso_set.update(country_data.get(lang_info["Country"], []))

            # Add iso codes from the same family
            iso_set.update(family_data.get(lang_info["Family"], []))

    return iso_set


def main():
    parser = argparse.ArgumentParser(description="Find data in NLLB languages given a list of ISO codes.")
    parser.add_argument("--directory", type=str, default=f"{SIL_NLP_ENV.mt_scripture_dir}", help=f"Directory to search. The default is {SIL_NLP_ENV.mt_scripture_dir}")
    parser.add_argument("iso_codes", type=str, nargs="+", help="List of ISO codes to search for")
    #parser.add_argument("--no_related", action='store_true', help="Only list specified languages and not related iso codes that are part of NLLB")

    args = parser.parse_args()
    iso_codes = args.iso_codes
    projects_folder = SIL_NLP_ENV.pt_projects_dir
    scripture_dir = Path(args.directory)

    print("Finding related languages and those spoken in the same country.")
    file_path = SIL_NLP_ENV.assets_dir / "languageFamilies.json"

    language_data, country_data, family_data = load_language_data(file_path)
    nllb_set = set(NLLB_ISOS)

    related_isos = process_iso_codes(iso_codes, language_data, country_data, family_data, NLLB_ISOS)

    if related_isos:
        # Remove iso codes not in NLLB    
        related_isos_in_nllb = sorted(related_isos.intersection(nllb_set))
        if related_isos_in_nllb:

            # Look for scriptures in these languages too.
            iso_codes.extend(related_isos_in_nllb)
            
            print(f"Found {len(related_isos_in_nllb)} languages that from the same country or language family in NLLB.")
            for related_iso_in_nllb in related_isos_in_nllb:
                lang_info = language_data[related_iso_in_nllb]
                print(
                    f"{related_iso_in_nllb}: {lang_info['Name']}, {lang_info['Country']}, {lang_info['Family']}"
                )
    else:
        print(f"Didn't find any language that is related or spoken in the same country in NLLB.")
    

    matching_files = []
    for filepath in scripture_dir.iterdir():
        if filepath.suffix == ".txt":
            iso_code = filepath.stem.split("-")[0]
            if iso_code in args.iso_codes:
                matching_files.append(filepath.stem)  # Remove .txt extension

    if matching_files:
        print("Matching files:")
        for file in matching_files:
            print(f"      - {file}")

        for file in matching_files:
            parts = file.split("-", maxsplit=1)
            if len(parts) > 1:
                iso = parts[0]
                project = parts[1]
                project_dir = projects_folder / project
                print(f"{project} exists: {project_dir.is_dir()}")
            else:
                print(f"Couldn't split {file} on '-'")
    else:
        print("No matching files found.")


if __name__ == "__main__":
    main()


    # projects_folder = Path("S:\Paratext\projects")
    # scripture_dir = args.directory
    # matching_files = []
    # for filename in os.listdir(scripture_dir):
    #     if filename.endswith(".txt"):
    #         iso_code = filename.split("-")[0]
    #         if iso_code in args.iso_codes:
    #             matching_files.append(os.path.splitext(filename)[0])  # Remove .txt extension

    # if matching_files:
    #     print("Matching files:")
    #     for file in matching_files:
    #         print(f"      - {file}")

    #     for file in matching_files:
    #         parts = file.split("-", maxsplit=1)
    #         if len(parts) > 1:
    #             iso = parts[0]
    #             project = parts[1]
    #             project_dir = projects_folder / project
    #             print(f"{project} exists: {project_dir.is_dir()}")
    #         else:
    #             print(f"Couldn't split {file} on '-'")
    # else:
    #     print("No matching files found.")

