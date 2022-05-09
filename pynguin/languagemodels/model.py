#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#

class _OpenAILanguageModel:
    """
    An interface for an OpenAI language model to generate/mutate tests as natural language.
    TODO(clemieux): starting by implementing a concrete instance of this.
    """

    def __init__(self):
        self._test_src : str
        self._authorization_key : str
        self._complete_model : str
        self._edit_model : str

    @property
    def test_src(self) -> str:
        """Provides the source of the module under test

        Returns:
            The source of the module under test
        """
        return self._test_src

    @test_src.setter
    def test_src(self, test_src: str):
        self._test_src = test_src


    @property
    def authorization_key(self) -> str:
        """Provides the authorization key used to query the model

        Returns:
            The organization id
        """
        return self._authorization_key


    @authorization_key.setter
    def authorization_key(self, authorization_key: str):
        self._authorization_key = authorization_key


    @property
    def complete_model(self) -> str:
        """Provides the name of the model used for completion tasks

        Returns:
            The name of the model used for completion tasks
        """
        return self._complete_model

    @complete_model.setter
    def complete_model(self, complete_model: str):
        self._complete_model = complete_model

    @property
    def edit_model(self) -> str:
        """Provides the name of the model used for editing tasks

        Returns:
            The name of the model used for editing tasks
        """
        return self._edit_model

    @edit_model.setter
    def edit_model(self, edit_model: str):
        self._edit_model = edit_model


languagemodel = _OpenAILanguageModel()
