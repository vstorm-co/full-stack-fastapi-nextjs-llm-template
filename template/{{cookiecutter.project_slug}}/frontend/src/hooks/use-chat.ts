"use client";

import { useCallback, useRef, useState } from "react";
import { nanoid } from "nanoid";
import { useWebSocket } from "./use-websocket";
import { useChatStore } from "@/stores";
import type { ChatMessage, ToolCall, WSEvent } from "@/types";
import { WS_URL } from "@/lib/constants";
{%- if cookiecutter.enable_conversation_persistence and cookiecutter.use_database %}
import { useConversationStore } from "@/stores";
{%- endif %}

{%- if cookiecutter.enable_conversation_persistence and cookiecutter.use_database %}
interface UseChatOptions {
  conversationId?: string | null;
  onConversationCreated?: (conversationId: string) => void;
}

export function useChat(options: UseChatOptions = {}) {
  const { conversationId, onConversationCreated } = options;
  const { setCurrentConversationId } = useConversationStore();
{%- else %}
export function useChat() {
{%- endif %}
  const {
    messages,
    addMessage,
    updateMessage,
    addToolCall,
    updateToolCall,
    clearMessages,
  } = useChatStore();

  const [isProcessing, setIsProcessing] = useState(false);
  const [currentMessageId, setCurrentMessageId] = useState<string | null>(null);
  // Use ref for groupId to avoid React state timing issues with rapid WebSocket events
  const currentGroupIdRef = useRef<string | null>(null);

  const handleWebSocketMessage = useCallback(
    (event: MessageEvent) => {
      const wsEvent: WSEvent = JSON.parse(event.data);

      // Helper to create a new message
      const createNewMessage = (content: string): string => {
        // Mark previous message as not streaming before creating new one
        if (currentMessageId) {
          updateMessage(currentMessageId, (msg) => ({
            ...msg,
            isStreaming: false,
          }));
        }

        const newMsgId = nanoid();
        addMessage({
          id: newMsgId,
          role: "assistant",
          content,
          timestamp: new Date(),
          isStreaming: true,
          toolCalls: [],
          groupId: currentGroupIdRef.current || undefined,
        });
        setCurrentMessageId(newMsgId);
        return newMsgId;
      };

      switch (wsEvent.type) {
{%- if cookiecutter.enable_conversation_persistence and cookiecutter.use_database %}
        case "conversation_created": {
          // Handle new conversation created by backend
          const { conversation_id } = wsEvent.data as { conversation_id: string };
          setCurrentConversationId(conversation_id);
          onConversationCreated?.(conversation_id);
          break;
        }

        case "message_saved": {
          // Message was saved to database, update local ID if needed
          // We don't need to do anything special here for now
          break;
        }
{%- endif %}

        case "model_request_start": {
          // PydanticAI/LangChain - create message immediately
          createNewMessage("");
          break;
        }

        case "crew_start":
        case "crew_started": {
          // CrewAI - generate groupId for this execution, wait for agent events
          currentGroupIdRef.current = nanoid();
          break;
        }

        case "text_delta": {
          // Append text delta to current message
          if (currentMessageId) {
            const content = (wsEvent.data as { index: number; content: string }).content;
            updateMessage(currentMessageId, (msg) => ({
              ...msg,
              content: msg.content + content,
            }));
          }
          break;
        }

        // CrewAI agent events - each agent gets its own message container
        case "agent_started": {
          const { agent } = wsEvent.data as {
            agent: string;
            task: string;
          };
          // Create NEW message for this agent (groupId read from ref)
          createNewMessage(`🤖 **${agent}** is starting...`);
          break;
        }

        case "agent_completed": {
          // Finalize current agent's message with output
          if (currentMessageId) {
            const { agent, output } = wsEvent.data as {
              agent: string;
              output: string;
            };
            updateMessage(currentMessageId, (msg) => ({
              ...msg,
              content: `✅ **${agent}**\n\n${output}`,
              isStreaming: false,
            }));
          }
          break;
        }

        // CrewAI task events - create separate message for each task
        case "task_started": {
          const { description, agent } = wsEvent.data as {
            task_id: string;
            description: string;
            agent: string;
          };
          // Create NEW message for this task (groupId read from ref)
          createNewMessage(`📋 **Task** (${agent})\n\n${description}`);
          break;
        }

        case "task_completed": {
          // Finalize the task message
          if (currentMessageId) {
            const { output, agent } = wsEvent.data as {
              task_id: string;
              output: string;
              agent: string;
            };
            updateMessage(currentMessageId, (msg) => ({
              ...msg,
              content: `✅ **Task completed** (${agent})\n\n${output}`,
              isStreaming: false,
            }));
          }
          break;
        }

        // CrewAI tool events
        case "tool_started": {
          if (currentMessageId) {
            const { tool_name, tool_args, agent } = wsEvent.data as {
              tool_name: string;
              tool_args: string;
              agent: string;
            };
            const toolCall: ToolCall = {
              id: nanoid(),
              name: tool_name,
              args: { input: tool_args, agent },
              status: "running",
            };
            addToolCall(currentMessageId, toolCall);
          }
          break;
        }

        case "tool_finished": {
          // Tool finished - update last tool call status
          if (currentMessageId) {
            const { tool_name, tool_result } = wsEvent.data as {
              tool_name: string;
              tool_result: string;
              agent: string;
            };
            // Find and update the matching tool call
            updateMessage(currentMessageId, (msg) => {
              const toolCalls = msg.toolCalls || [];
              const lastToolCall = toolCalls.find(
                (tc) => tc.name === tool_name && tc.status === "running"
              );
              if (lastToolCall) {
                return {
                  ...msg,
                  toolCalls: toolCalls.map((tc) =>
                    tc.id === lastToolCall.id
                      ? { ...tc, result: tool_result, status: "completed" as const }
                      : tc
                  ),
                };
              }
              return msg;
            });
          }
          break;
        }

        // LLM events (can be used for showing thinking status)
        case "llm_started":
        case "llm_completed": {
          // LLM lifecycle events - optionally show status
          break;
        }

        case "tool_call": {
          // Add tool call to current message
          if (currentMessageId) {
            const { tool_name, args, tool_call_id } = wsEvent.data as {
              tool_name: string;
              args: Record<string, unknown>;
              tool_call_id: string;
            };
            const toolCall: ToolCall = {
              id: tool_call_id,
              name: tool_name,
              args,
              status: "running",
            };
            addToolCall(currentMessageId, toolCall);
          }
          break;
        }

        case "tool_result": {
          // Update tool call with result
          if (currentMessageId) {
            const { tool_call_id, content } = wsEvent.data as {
              tool_call_id: string;
              content: string;
            };
            updateToolCall(currentMessageId, tool_call_id, {
              result: content,
              status: "completed",
            });
          }
          break;
        }

        case "final_result": {
          // Finalize message
          if (currentMessageId) {
            const { output } = wsEvent.data as { output: string };
            if (output) {
              updateMessage(currentMessageId, (msg) => ({
                ...msg,
                content: msg.content || output,
                isStreaming: false,
              }));
            } else {
              updateMessage(currentMessageId, (msg) => ({
                ...msg,
                isStreaming: false,
              }));
            }
          }
          setIsProcessing(false);
          setCurrentMessageId(null);
          currentGroupIdRef.current = null;
          break;
        }

        case "error": {
          // Handle error
          if (currentMessageId) {
            const { message } = wsEvent.data as { message: string };
            updateMessage(currentMessageId, (msg) => ({
              ...msg,
              content: msg.content + `\n\n❌ Error: ${message || "Unknown error"}`,
              isStreaming: false,
            }));
          }
          setIsProcessing(false);
          break;
        }

        case "complete": {
          setIsProcessing(false);
          break;
        }
      }
    },
{%- if cookiecutter.enable_conversation_persistence and cookiecutter.use_database %}
    [currentMessageId, addMessage, updateMessage, addToolCall, updateToolCall, setCurrentConversationId, onConversationCreated]
{%- else %}
    [currentMessageId, addMessage, updateMessage, addToolCall, updateToolCall]
{%- endif %}
  );

  const wsUrl = `${WS_URL}/api/v1/ws/agent`;

  const { isConnected, connect, disconnect, sendMessage } = useWebSocket({
    url: wsUrl,
    onMessage: handleWebSocketMessage,
  });

  const sendChatMessage = useCallback(
    (content: string) => {
      // Add user message
      const userMessage: ChatMessage = {
        id: nanoid(),
        role: "user",
        content,
        timestamp: new Date(),
      };
      addMessage(userMessage);

      // Send to WebSocket
      setIsProcessing(true);
{%- if cookiecutter.enable_conversation_persistence and cookiecutter.use_database %}
      sendMessage({
        message: content,
        conversation_id: conversationId || null,
      });
{%- else %}
      sendMessage({ message: content });
{%- endif %}
    },
{%- if cookiecutter.enable_conversation_persistence and cookiecutter.use_database %}
    [addMessage, sendMessage, conversationId]
{%- else %}
    [addMessage, sendMessage]
{%- endif %}
  );

  return {
    messages,
    isConnected,
    isProcessing,
    connect,
    disconnect,
    sendMessage: sendChatMessage,
    clearMessages,
  };
}
