from ..data_models.query import Query


class Component:
    """
    Base class for all components in the pipeline.
    """

    def run(self, queries: list[Query]) -> list[Query]:
        """Runs the component on the provided queries.

        Args:
            queries (list[Query]): A list of queries to process.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Returns:
            list[Query]: The processed queries.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def name(self) -> str:
        """Returns the name of the component.

        Returns:
            str: The name of the component.
        """
        return self.__class__.__name__
