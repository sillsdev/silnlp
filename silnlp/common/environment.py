import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Root directory
gutenberg_path = os.getenv("GUTENBERG_PATH")
nlpDir = Path(gutenberg_path if gutenberg_path is not None else r"G:/Shared drives/Gutenberg")

# API.Bible directories
apiBibleDir = nlpDir / r"API.Bible"
apiBibleAnalysisDir = apiBibleDir / r"Analysis"

# Bible Technologies.net directories
bibleTechDir = nlpDir / r"Bible Technologies.net"
bibleTechAnalysisDir = bibleTechDir / r"Analysis"

# Christodoulopoulos directories
christoDir = nlpDir / r"Christodoulopoulos"
christoResourceDir = christoDir / r"resources.CES"
christoTextDir = christoDir / r"resources.txt"
christoAnalysisDir = christoDir / r"Analysis"

# Cysouw directories
cysouwDir = nlpDir / r"Cysouw"
cysouwAnalysisDir = cysouwDir / r"Analysis"

# DBL directories
dblDir = nlpDir / r"DBL"
dblAnalysisDir = dblDir / r"Analysis"

# Paratext directories
paratextDir = nlpDir / r"Paratext"
paratextUnzippedDir = paratextDir / r"Paratext.unzipped"
paratextRippedDir = paratextDir / r"Paratext.ripped"
paratextAnalysisDir = paratextDir / r"Analysis"
paratextPreprocessedDir = paratextDir / r"Paratext.preprocessed"

# Reference data directories
refDataDir = nlpDir / r"Reference Data"
refDataEthnologueDir = refDataDir / r"Ethnologue"
refDataIso639Dir = refDataDir / r"ISO639"
refDataParatextDir = refDataDir / r"Paratext"

# Scripture API directories
scriptureApiDir = nlpDir / r"Scripture API"
scriptureApiAnalysisDir = scriptureApiDir / r"Analysis"

paratext_name = os.getenv("PARATEXT_NAME")
paratext_password = os.getenv("PARATEXT_PASSWORD")

dbl_name = os.getenv("DBL_NAME")
dbl_password = os.getenv("DBL_PASSWORD")

scripture_api_key = os.getenv("SCRIPTURE_API_KEY")
