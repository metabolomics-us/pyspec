from abc import ABC, abstractmethod


class Share(ABC):
    """
    provides an uniform way to load and share datasets between clients and locations
    """

    @abstractmethod
    def retrieve(self, name: str, root_folder: str = 'datasets', force:bool = False):
        """
        retrieves the given dataset and stores it at the given location
        :param name: 
        :param root: 
        :param force: do we want to overwrite local data
        :return: 
        """

    @abstractmethod
    def submit(self, name: str, root_folder: str = 'datasets'):
        """
        uploads the given dataset to the remote location. If it already exists, it will be replaced
        :param name: 
        :return: 
        """

    @abstractmethod
    def exists(self, name: str) -> bool:
        """
        does this dataset exist remotely
        :param name: 
        :return: 
        """
