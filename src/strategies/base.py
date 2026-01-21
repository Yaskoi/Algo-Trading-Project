from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def on_bar(self, t, row, history):
        """
        Returns target_position in {-1,0,1} (baseline) or float sizing.
        history: df.iloc[:current_index+1]
        """
        raise NotImplementedError
