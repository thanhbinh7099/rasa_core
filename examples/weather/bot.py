from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging

import numpy as np

import sys
import warnings

from examples.weather.md2json import convert
from rasa_core.actions import Action
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy

logger = logging.getLogger(__name__)



class ActionSearchWeather(Action):
    def name(self):
        return 'action_search_weather'

    def run(self, dispatcher, tracker, domain):
        str = ['Nắng', 'Mưa', 'Bão', 'Lạnh', 'Âm u', 'Tận thế']
        dispatcher.utter_message("Đợi chút. Thời tiết ngày hôm đó là " + str[np.random.randint(6)])
        return []


class ActionSuggest(Action):

    def name(self):
        return 'action_suggest'

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message("Good day")
        return []

class WeatherPolicy(KerasPolicy):
    def model_architecture(self, num_features, num_actions, max_history_len):
        """Build a Keras model and return a compiled model."""
        from keras.layers import LSTM, Activation, Masking, Dense
        from keras.models import Sequential

        n_hidden = 32  # size of hidden layer in LSTM
        # Build Model
        batch_shape = (None, max_history_len, num_features)

        model = Sequential()
        model.add(Masking(-1, batch_input_shape=batch_shape))
        model.add(LSTM(n_hidden, batch_input_shape=batch_shape))
        model.add(Dense(input_dim=n_hidden, output_dim=num_actions))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        logger.debug(model.summary())
        return model


def train_dialogue(domain_file="weather_domain.yml",
                   model_path="models/dialogue",
                   training_data_file="data/babi_stories.md"):
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(), WeatherPolicy()])

    agent.train(
            training_data_file,
            max_history=3,
            epochs=2000,
            batch_size=50,
            augmentation_factor=50,
            validation_split=0.2
    )

    agent.persist(model_path)
    return agent


def train_nlu():
    from rasa_nlu.converters import load_data
    from rasa_nlu.config import RasaNLUConfig
    from rasa_nlu.model import Trainer

    convert('data/thoi_tiet.md','data/thoi_tiet.json')

    training_data = load_data('data/thoi_tiet.json')
    trainer = Trainer(RasaNLUConfig("domain-config.json"))
    trainer.train(training_data)
    model_directory = trainer.persist('models/nlu/', fixed_model_name="current")

    return model_directory


def run(serve_forever=True):
    agent = Agent.load("models/dialogue",
                       interpreter=RasaNLUInterpreter("models/nlu/default/current"))

    if serve_forever:
        agent.handle_channel(ConsoleInputChannel())
    return agent

if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")

    parser = argparse.ArgumentParser(
            description='starts the bot')

    parser.add_argument(
            'task',
            choices=["train-nlu", "train-dialogue", "run"],
            help="what the bot should do - e.g. run or train?")
    task = parser.parse_args().task

    # decide what to do based on first parameter of the script
    if task == "train-nlu":
        train_nlu()
    elif task == "train-dialogue":
        train_dialogue()
    elif task == "run":
        run()
    else:
        warnings.warn("Need to pass either 'train-nlu', 'train-dialogue' or "
                      "'run' to use the script.")
        exit(1)
