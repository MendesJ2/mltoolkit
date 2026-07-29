from abc import ABC
import logging


class BaseComponent(ABC):
    """
    Base class for every component of the framework.
    """

    def __init__(self, config=None, logger=None):

        self.config = config

        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @property
    def name(self):

        return self.__class__.__name__

    def info(self, message):

        self.logger.info(f"[{self.name}] {message}")

    def warning(self, message):

        self.logger.warning(f"[{self.name}] {message}")

    def error(self, message):

        self.logger.error(f"[{self.name}] {message}")
