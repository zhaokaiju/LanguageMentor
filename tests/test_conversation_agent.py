import sys
import os
import unittest
from unittest.mock import patch, MagicMock, mock_open

# 将 src 目录添加到模块搜索路径，方便导入项目中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from agents.conversation_agent import ConversationAgent


class TestConversationAgent(unittest.TestCase):

    def setUp(self):
        """
        在每个测试方法运行前执行，初始化 ConversationAgent 实例和测试数据。
        """
        self.session_id = "test_session"

        self.agent = ConversationAgent(session_id=self.session_id)

    def test_initialization(self):
        """测试代理初始化是否正确"""
        self.assertEqual(self.agent.name, "conversation")
        self.assertEqual(self.agent.prompt_file, "prompts/conversation_prompt.txt")
        self.assertEqual(self.agent.session_id, "test_session")
        self.assertEqual(self.agent.prompt, "Test system prompt")
        self.assertIsNone(self.agent.intro_file)
        self.assertEqual(self.agent.intro_messages, [])
        self.mock_file.assert_called_once_with("prompts/conversation_prompt.txt", "r", encoding="utf-8")

    def test_load_prompt_file_not_found(self):
        """测试提示文件不存在时的异常处理"""
        self.mock_file.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            AgentBase("test", "invalid_prompt.txt")

    @patch("json.load", return_value=[{"role": "system", "content": "Intro message"}])
    def test_load_intro(self, mock_json):
        """测试加载初始消息文件"""
        with patch("builtins.open", mock_open(read_data='[{"role": "system", "content": "Intro message"}]')):
            agent = AgentBase("test", "prompt.txt", intro_file="intro.json")
            self.assertEqual(agent.intro_messages, [{"role": "system", "content": "Intro message"}])

    @patch.object(AgentBase, "create_chatbot")
    def test_session_id_default(self, mock_create_chatbot):
        """测试默认会话ID生成"""
        agent = ConversationAgent()
        self.assertEqual(agent.session_id, "conversation")

    @patch("langchain_core.runnables.history.RunnableWithMessageHistory.invoke")
    @patch("langchain_core.messages.HumanMessage")
    def test_chat_with_history(self, mock_human_message, mock_invoke):
        """测试带历史记录的对话功能"""
        # 配置模拟对象
        mock_response = MagicMock()
        mock_response.content = "模拟回复内容"
        mock_invoke.return_value = mock_response

        # 执行测试
        response = self.agent.chat_with_history("用户输入")

        # 验证调用参数
        mock_human_message.assert_called_once_with(content="用户输入")
        mock_invoke.assert_called_once_with(
            [mock_human_message.return_value],
            {"configurable": {"session_id": "test_session"}}
        )
        self.assertEqual(response, "模拟回复内容")

    @patch("langchain_core.runnables.history.RunnableWithMessageHistory.invoke")
    def test_custom_session_id(self, mock_invoke):
        """测试自定义会话ID功能"""
        # 创建新代理
        custom_agent = ConversationAgent(session_id="custom_session")

        # 调用方法
        custom_agent.chat_with_history("输入", session_id="another_session")

        # 验证使用的会话ID
        mock_invoke.assert_called_once()
        call_args = mock_invoke.call_args[1]
        self.assertEqual(call_args["configurable"]["session_id"], "another_session")


if __name__ == '__main__':
    unittest.main()
