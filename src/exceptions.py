class RepositoryException(Exception):
    """Base exception for repository-related errors."""



class PaperNotSaved(RepositoryException):
    """Exception raised when paper data is not saved."""


class ParsingException(Exception):
    """Base exception for parsing-related errors."""


# Week 2: PDF parsing exceptions (implemented)
class PDFParsingException(ParsingException):
    """Base exception for PDF parsing-related errors."""


class PDFValidationError(PDFParsingException):
    """Exception raised when PDF file validation fails."""


# Week 2+: Metadata fetching exceptions
class MetadataFetchingException(Exception):
    """Base exception for metadata fetching pipeline errors."""


class PipelineException(MetadataFetchingException):
    """Exception raised during pipeline execution."""


class LLMException(Exception):
    """Base exception for LLM-related errors."""




# General application exceptions
class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""
