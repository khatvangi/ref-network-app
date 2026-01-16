# providers module
from .base import PaperProvider, PaperStub
from .openalex import OpenAlexProvider
from .semantic_scholar import SemanticScholarProvider
from .pubmed import PubMedProvider
from .crossref import CrossrefProvider
from .aggregator import ProviderAggregator, AggregatorConfig

# google scholar is optional
try:
    from .google_scholar import GoogleScholarProvider
except ImportError:
    GoogleScholarProvider = None

__all__ = [
    'PaperProvider',
    'PaperStub',
    'OpenAlexProvider',
    'SemanticScholarProvider',
    'PubMedProvider',
    'CrossrefProvider',
    'GoogleScholarProvider',
    'ProviderAggregator',
    'AggregatorConfig'
]
