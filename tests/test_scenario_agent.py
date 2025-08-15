import sys
import os
import unittest
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open

# 将 src 目录添加到模块搜索路径，方便导入项目中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from agents.scenario_agent import ScenarioAgent

class TestScenarioAgent(unittest.TestCase):

    def setUp(self):
        # 创建临时目录结构
        self.temp_dir = tempfile.mkdtemp()
        self.prompts_dir = os.path.join(self.temp_dir, "prompts")
        self.content_dir = os.path.join(self.temp_dir, "content", "intro")
        os.makedirs(self.prompts_dir)
        os.makedirs(self.content_dir)

        # 创建测试提示文件
        self.scenario_name = "fantasy_adventure"
        self.prompt_file = os.path.join(self.prompts_dir, f"{self.scenario_name}_prompt.txt")
        with open(self.prompt_file, "w") as f:
            f.write("你是一个奇幻冒险助手")

        # 创建测试介绍文件
        self.intro_file = os.path.join(self.content_dir, f"{self.scenario_name}.json")
        with open(self.intro_file, "w") as f:
            json.dump(["欢迎来到奇幻世界！", "准备好开始冒险了吗？"], f)

        # 导入被测试类
        self.ScenarioAgent = ScenarioAgent

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.temp_dir)

    @patch('langchain_core.runnables.history.RunnableWithMessageHistory')
    @patch('langchain_community.chat_models.ChatTongyi')
    def test_initialization(self, mock_chat, mock_runnable):
        """测试代理正确初始化"""
        agent = self.ScenarioAgent(self.scenario_name)

        # 验证属性设置
        self.assertEqual(agent.name, self.scenario_name)
        self.assertEqual(agent.session_id, self.scenario_name)
        self.assertEqual(agent.prompt, "你是一个奇幻冒险助手")
        self.assertEqual(agent.intro_messages, ["欢迎来到奇幻世界！", "准备好开始冒险了吗？"])

        # 验证文件路径
        self.assertEqual(agent.prompt_file, self.prompt_file)
        self.assertEqual(agent.intro_file, self.intro_file)

    @patch('langchain_core.runnables.history.RunnableWithMessageHistory')
    @patch('langchain_community.chat_models.ChatTongyi')
    def test_custom_session_id(self, mock_chat, mock_runnable):
        """测试自定义会话ID"""
        custom_id = "user_123_fantasy"
        agent = self.ScenarioAgent(self.scenario_name, session_id=custom_id)
        self.assertEqual(agent.session_id, custom_id)

    @patch('agents.scenario_agent.get_session_history')
    @patch('langchain_core.runnables.history.RunnableWithMessageHistory')
    @patch('langchain_community.chat_models.ChatTongyi')
    def test_chat_with_history(self, mock_chat, mock_runnable, mock_get_history):
        """测试带历史记录的聊天功能"""
        # 配置模拟对象
        mock_invoke = MagicMock()
        mock_invoke.content = "前方发现巨龙洞穴！"
        mock_runnable.return_value.invoke = MagicMock(return_value=mock_invoke)

        agent = self.ScenarioAgent(self.scenario_name)
        response = agent.chat_with_history("我应该去哪里探险？")

        # 验证调用参数
        args, kwargs = mock_runnable.return_value.invoke.call_args
        self.assertIsInstance(args[0][0], HumanMessage)
        self.assertEqual(args[0][0].content, "我应该去哪里探险？")
        self.assertEqual(kwargs["configurable"]["session_id"], self.scenario_name)

        # 验证响应
        self.assertEqual(response, "前方发现巨龙洞穴！")

    @patch('langchain_core.messages.AIMessage')
    @patch('agents.scenario_agent.get_session_history')
    @patch('random.choice')
    @patch('langchain_core.runnables.history.RunnableWithMessageHistory')
    @patch('langchain_community.chat_models.ChatTongyi')
    def test_start_new_session_empty_history(
            self, mock_chat, mock_runnable, mock_choice, mock_get_history, mock_ai_message
    ):
        """测试新会话初始化（空历史）"""
        # 模拟空历史
        mock_history = MagicMock()
        mock_history.messages = []
        mock_get_history.return_value = mock_history

        # 模拟随机选择
        mock_choice.return_value = "欢迎来到奇幻世界！"

        agent = self.ScenarioAgent(self.scenario_name)
        response = agent.start_new_session()

        # 验证添加了初始消息
        mock_history.add_message.assert_called_once()
        self.assertEqual(response, "欢迎来到奇幻世界！")

    @patch('agents.scenario_agent.get_session_history')
    @patch('langchain_core.runnables.history.RunnableWithMessageHistory')
    @patch('langchain_community.chat_models.ChatTongyi')
    def test_start_new_session_existing_history(self, mock_chat, mock_runnable, mock_get_history):
        """测试已有历史记录的会话初始化"""
        # 模拟已有历史
        mock_history = MagicMock()
        mock_history.messages = [
            MagicMock(content="第一条消息"),
            MagicMock(content="上一条消息")
        ]
        mock_get_history.return_value = mock_history

        agent = self.ScenarioAgent(self.scenario_name)
        response = agent.start_new_session()

        # 验证没有添加新消息
        mock_history.add_message.assert_not_called()
        self.assertEqual(response, "上一条消息")

    @patch('langchain_core.runnables.history.RunnableWithMessageHistory')
    @patch('langchain_community.chat_models.ChatTongyi')
    def test_missing_prompt_file(self, mock_chat, mock_runnable):
        """测试缺少提示文件的情况"""
        os.remove(self.prompt_file)
        with self.assertRaises(FileNotFoundError):
            self.ScenarioAgent(self.scenario_name)

    @patch('langchain_core.runnables.history.RunnableWithMessageHistory')
    @patch('langchain_community.chat_models.ChatTongyi')
    def test_missing_intro_file(self, mock_chat, mock_runnable):
        """测试缺少介绍文件的情况"""
        os.remove(self.intro_file)
        with self.assertRaises(FileNotFoundError):
            self.ScenarioAgent(self.scenario_name)

    @patch('langchain_core.runnables.history.RunnableWithMessageHistory')
    @patch('langchain_community.chat_models.ChatTongyi')
    def test_invalid_intro_file(self, mock_chat, mock_runnable):
        """测试无效的介绍文件"""
        with open(self.intro_file, "w") as f:
            f.write("无效的JSON内容")

        with self.assertRaises(ValueError):
            self.ScenarioAgent(self.scenario_name)


if __name__ == "__main__":
    unittest.main()
