/**
 * 类人脑双系统AI架构 - API接口
 * Human-Like Brain Dual-System AI Architecture - API Routes
 */

import { NextRequest, NextResponse } from 'next/server';
import { brainArchitecture } from '../../../brain-architecture/core';

// 初始化标志
let isInitialized = false;

// 确保初始化
async function ensureInitialized() {
  if (!isInitialized) {
    const result = await brainArchitecture.initialize();
    if (result.success) {
      isInitialized = true;
    }
    return result;
  }
  return { success: true, message: 'Already initialized' };
}

// GET请求 - 获取架构状态
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const action = searchParams.get('action');

  try {
    await ensureInitialized();

    switch (action) {
      case 'state':
        return NextResponse.json({
          success: true,
          data: brainArchitecture.getState(),
        });

      case 'metrics':
        return NextResponse.json({
          success: true,
          data: {
            performance: brainArchitecture.getPerformanceMetrics(),
            memory: brainArchitecture.getMemoryStatistics(),
            stdp: brainArchitecture.getSTDPStatistics(),
            optimizationMode: brainArchitecture.getCurrentOptimizationMode(),
          },
        });

      case 'scenarios':
        return NextResponse.json({
          success: true,
          data: brainArchitecture.getAvailableScenarios(),
        });

      case 'evaluate':
        const report = await brainArchitecture.evaluate();
        return NextResponse.json({
          success: true,
          data: report,
        });

      default:
        return NextResponse.json({
          success: true,
          data: {
            isReady: brainArchitecture.isReady(),
            isRunning: brainArchitecture.isRunning(),
            state: brainArchitecture.getState(),
          },
        });
    }
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

// POST请求 - 执行操作
export async function POST(request: NextRequest) {
  try {
    const initResult = await ensureInitialized();
    if (!initResult.success) {
      return NextResponse.json(
        { success: false, error: initResult.message },
        { status: 500 }
      );
    }

    const body = await request.json();
    const { action, data } = body;

    switch (action) {
      case 'start':
        brainArchitecture.start();
        return NextResponse.json({
          success: true,
          message: 'Architecture started',
        });

      case 'stop':
        brainArchitecture.stop();
        return NextResponse.json({
          success: true,
          message: 'Architecture stopped',
        });

      case 'infer':
        if (!data?.input) {
          return NextResponse.json(
            { success: false, error: 'Input is required' },
            { status: 400 }
          );
        }
        const result = await brainArchitecture.infer(data.input);
        return NextResponse.json({
          success: true,
          data: result,
        });

      case 'setMode':
        if (!data?.mode) {
          return NextResponse.json(
            { success: false, error: 'Mode is required' },
            { status: 400 }
          );
        }
        brainArchitecture.setOptimizationMode(data.mode);
        return NextResponse.json({
          success: true,
          message: `Mode set to ${data.mode}`,
        });

      case 'train':
        if (!data?.scenario) {
          return NextResponse.json(
            { success: false, error: 'Scenario is required' },
            { status: 400 }
          );
        }
        const trainResult = await brainArchitecture.trainScenario(
          data.scenario,
          data.epochs
        );
        return NextResponse.json({
          success: trainResult.success,
          data: trainResult,
        });

      case 'evaluate':
        const report = await brainArchitecture.evaluate();
        return NextResponse.json({
          success: true,
          data: report,
        });

      default:
        return NextResponse.json(
          { success: false, error: 'Unknown action' },
          { status: 400 }
        );
    }
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
