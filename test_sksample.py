# Copyright (c) Microsoft. All rights reserved.

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestChatCompletionAgent:
    """Tests for the ChatCompletionAgent in sksample.py"""

    @pytest.fixture
    def mock_azure_credential(self):
        """Mock AzureCliCredential"""
        with patch("sksample.AzureCliCredential") as mock:
            yield mock

    @pytest.fixture
    def mock_azure_chat_completion(self):
        """Mock AzureChatCompletion service"""
        with patch("sksample.AzureChatCompletion") as mock:
            yield mock

    @pytest.fixture
    def mock_chat_completion_agent(self):
        """Mock ChatCompletionAgent"""
        with patch("sksample.ChatCompletionAgent") as mock:
            # Create a mock response
            mock_response = MagicMock()
            mock_response.name = "Assistant"
            mock_response.__str__ = lambda self: "This is a test response."
            
            # Setup the agent instance
            agent_instance = MagicMock()
            agent_instance.get_response = AsyncMock(return_value=mock_response)
            mock.return_value = agent_instance
            yield mock

    @pytest.mark.asyncio
    async def test_main_creates_agent_with_correct_parameters(
        self, mock_azure_credential, mock_azure_chat_completion, mock_chat_completion_agent
    ):
        """Test that main() creates an agent with the correct configuration"""
        from sksample import main

        await main()

        # Verify ChatCompletionAgent was created
        mock_chat_completion_agent.assert_called_once()
        call_kwargs = mock_chat_completion_agent.call_args.kwargs
        
        assert call_kwargs["name"] == "Assistant"
        assert call_kwargs["instructions"] == "Answer questions about the world in one sentence."

    @pytest.mark.asyncio
    async def test_main_invokes_agent_for_each_user_input(
        self, mock_azure_credential, mock_azure_chat_completion, mock_chat_completion_agent
    ):
        """Test that main() invokes the agent for each user input"""
        from sksample import main, USER_INPUTS

        await main()

        agent_instance = mock_chat_completion_agent.return_value
        
        # Verify get_response was called for each user input
        assert agent_instance.get_response.call_count == len(USER_INPUTS)

    @pytest.mark.asyncio
    async def test_main_passes_correct_messages_to_agent(
        self, mock_azure_credential, mock_azure_chat_completion, mock_chat_completion_agent
    ):
        """Test that main() passes the correct messages to the agent"""
        from sksample import main, USER_INPUTS

        await main()

        agent_instance = mock_chat_completion_agent.return_value
        
        # Verify each user input was passed to get_response
        calls = agent_instance.get_response.call_args_list
        for i, call in enumerate(calls):
            assert call.kwargs["messages"] == USER_INPUTS[i]

    @pytest.mark.asyncio
    async def test_azure_chat_completion_configuration(
        self, mock_azure_credential, mock_azure_chat_completion, mock_chat_completion_agent
    ):
        """Test that AzureChatCompletion is configured correctly"""
        from sksample import main

        await main()

        # Verify AzureChatCompletion was created with expected parameters
        mock_azure_chat_completion.assert_called_once()
        call_kwargs = mock_azure_chat_completion.call_args.kwargs
        
        assert "credential" in call_kwargs
        assert "endpoint" in call_kwargs
        assert "deployment_name" in call_kwargs
        assert "api_version" in call_kwargs


class TestUserInputs:
    """Tests for USER_INPUTS constant"""

    def test_user_inputs_is_list(self):
        """Test that USER_INPUTS is a list"""
        from sksample import USER_INPUTS
        assert isinstance(USER_INPUTS, list)

    def test_user_inputs_has_expected_questions(self):
        """Test that USER_INPUTS contains the expected questions"""
        from sksample import USER_INPUTS
        
        assert "Why is the sky blue?" in USER_INPUTS
        assert "What is the capital of France?" in USER_INPUTS

    def test_user_inputs_are_strings(self):
        """Test that all USER_INPUTS are strings"""
        from sksample import USER_INPUTS
        
        for user_input in USER_INPUTS:
            assert isinstance(user_input, str)
