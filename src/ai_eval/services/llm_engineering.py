from abc import ABC, abstractmethod
from typing import Any

class LLMEngineering(ABC):
    def __init__(self, verbose: bool = False):
        """Generic read/write data to the langfuse service
        Args:
           verbose (bool, optional): should user information be displayed? Defaults to False
        """
        self.verbose = verbose
    
    @abstractmethod
    def createDataset(self, **kwargs: Any) -> Any:
        """Abstract method to create dataset"""
        raise NotImplementedError("Subclasses must implement createDataset method")

    @abstractmethod
    def read(self, **kwargs: Any) -> Any:
        """Abstract method to read dataset"""
        raise NotImplementedError("Subclasses must implement read method")

    @abstractmethod
    def insertItems(self, **kwargs: Any) -> Any:
        """Abstract method to insert items into dataset"""
        raise NotImplementedError("Subclasses must implement insertItems method")
    
    @abstractmethod
    def deleteDataset(self, **kwargs: Any) -> Any:
        """Abstract method to delete dataset"""
        raise NotImplementedError("Subclasses must implement deleteDataset method")

    @abstractmethod
    def update_from_pandas(self, **kwargs: Any) -> Any:
        """Abstract method to update from pandas"""
        raise NotImplementedError("Subclasses must implement update_from_pandas method")

    @abstractmethod
    def createPrompt(self, **kwargs: Any) -> Any:
        """Abstract method to create prompt"""
        raise NotImplementedError("Subclasses must implement createPrompt method")