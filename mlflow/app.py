import gradio as gr
import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from typing import List, Dict, Any

class MLflowTracker:
    def __init__(self, tracking_uri: str = None):
        """MLflow 추적 클래스 초기화"""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

    def get_experiments(self) -> List[Dict]:
        """모든 실험 목록 가져오기"""
        experiments = self.client.search_experiments()
        return [{"name": exp.name, "id": exp.experiment_id} for exp in experiments]

    def get_runs_by_experiment(self, experiment_id: str, tags: List[str] = None) -> pd.DataFrame:
        """실험별 실행 데이터 가져오기"""
        filter_string = ""
        if tags:
            tag_filters = [f"tags.{tag} != ''" for tag in tags]
            filter_string = " AND ".join(tag_filters)

        runs = self.client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"]
        )

        data = []
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or "Unnamed",
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                "end_time": datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
            }

            # 메트릭 추가
            for metric_key, metric_value in run.data.metrics.items():
                run_data[f"metric_{metric_key}"] = metric_value

            # 파라미터 추가
            for param_key, param_value in run.data.params.items():
                run_data[f"param_{param_key}"] = param_value

            # 태그 추가
            for tag_key, tag_value in run.data.tags.items():
                run_data[f"tag_{tag_key}"] = tag_value

            data.append(run_data)

        return pd.DataFrame(data)

    def get_run_metrics_history(self, run_id: str, metric_name: str) -> pd.DataFrame:
        """특정 실행의 메트릭 히스토리 가져오기"""
        history = self.client.get_metric_history(run_id, metric_name)
        data = [{"step": m.step, "value": m.value, "timestamp": datetime.fromtimestamp(m.timestamp / 1000)}
                for m in history]
        return pd.DataFrame(data)

    def create_experiment(self, name: str, tags: Dict[str, str] = None) -> str:
        """새 실험 생성"""
        experiment_id = mlflow.create_experiment(name, tags=tags)
        return experiment_id

    def add_tag_to_run(self, run_id: str, key: str, value: str):
        """실행에 태그 추가"""
        self.client.set_tag(run_id, key, value)

    def log_command(self, run_id: str, command: str):
        """실행 명령어 로깅"""
        self.client.set_tag(run_id, "command", command)
        self.client.set_tag(run_id, "logged_at", datetime.now().isoformat())

# MLflow 추적기 인스턴스 생성
tracker = MLflowTracker()

def refresh_experiments():
    """실험 목록 새로고침"""
    experiments = tracker.get_experiments()
    choices = [f"{exp['name']} (ID: {exp['id']})" for exp in experiments]
    return gr.update(choices=choices, value=choices[0] if choices else None)

def load_experiment_data(experiment_selection, tag_filter):
    """선택된 실험의 데이터 로드"""
    if not experiment_selection:
        return None, "실험을 선택해주세요."

    # 실험 ID 추출
    experiment_id = experiment_selection.split("ID: ")[1].split(")")[0]

    # 태그 필터 처리
    tags = [tag.strip() for tag in tag_filter.split(",") if tag.strip()] if tag_filter else None

    try:
        df = tracker.get_runs_by_experiment(experiment_id, tags)
        if df.empty:
            return None, "해당 조건의 실행이 없습니다."

        return df, f"총 {len(df)}개의 실행을 로드했습니다."
    except Exception as e:
        return None, f"오류 발생: {str(e)}"

def plot_loss_comparison(df, loss_columns):
    """손실 값 비교 플롯 생성"""
    if df is None or df.empty:
        return None

    # 손실 관련 컬럼 찾기
    available_loss_cols = [col for col in df.columns if any(loss_term in col.lower() for loss_term in ['loss', 'error', 'cost'])]

    if not available_loss_cols:
        return None

    # 선택된 컬럼들만 사용
    selected_cols = [col for col in loss_columns.split(",") if col.strip() in available_loss_cols] if loss_columns else available_loss_cols[:3]

    if not selected_cols:
        selected_cols = available_loss_cols[:3]

    fig = go.Figure()

    for col in selected_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['run_name'],
                y=df[col],
                mode='markers+lines',
                name=col.replace('metric_', ''),
                text=df['run_id'],
                hovertemplate=f'<b>{col}</b><br>Run: %{{text}}<br>Value: %{{y}}<extra></extra>'
            ))

    fig.update_layout(
        title="실행별 손실 값 비교",
        xaxis_title="실행 이름",
        yaxis_title="손실 값",
        hovermode='closest'
    )

    return fig

def plot_metrics_over_time(df):
    """시간에 따른 메트릭 변화 플롯"""
    if df is None or df.empty:
        return None

    metric_cols = [col for col in df.columns if col.startswith('metric_')]

    if not metric_cols:
        return None

    fig = px.scatter(df, x='start_time', y=metric_cols[0],
                     hover_data=['run_name', 'run_id'],
                     title=f"시간에 따른 {metric_cols[0]} 변화")

    return fig

def get_run_details(df, run_selection):
    """선택된 실행의 세부 정보 표시"""
    if df is None or df.empty or not run_selection:
        return "실행을 선택해주세요."

    try:
        run_idx = int(run_selection)
        if run_idx >= len(df):
            return "유효하지 않은 실행 인덱스입니다."

        run_data = df.iloc[run_idx]

        details = f"**실행 ID:** {run_data['run_id']}\n"
        details += f"**실행 이름:** {run_data['run_name']}\n"
        details += f"**상태:** {run_data['status']}\n"
        details += f"**시작 시간:** {run_data['start_time']}\n\n"

        # 메트릭 정보
        details += "**메트릭:**\n"
        for col in df.columns:
            if col.startswith('metric_'):
                metric_name = col.replace('metric_', '')
                details += f"- {metric_name}: {run_data[col]}\n"

        # 파라미터 정보
        details += "\n**파라미터:**\n"
        for col in df.columns:
            if col.startswith('param_'):
                param_name = col.replace('param_', '')
                details += f"- {param_name}: {run_data[col]}\n"

        # 태그 정보
        details += "\n**태그:**\n"
        for col in df.columns:
            if col.startswith('tag_'):
                tag_name = col.replace('tag_', '')
                details += f"- {tag_name}: {run_data[col]}\n"

        return details

    except Exception as e:
        return f"오류 발생: {str(e)}"

def add_tag_interface(df, run_selection, tag_key, tag_value):
    """실행에 태그 추가"""
    if df is None or df.empty or not run_selection or not tag_key or not tag_value:
        return "모든 필드를 입력해주세요."

    try:
        run_idx = int(run_selection)
        run_id = df.iloc[run_idx]['run_id']

        tracker.add_tag_to_run(run_id, tag_key, tag_value)
        return f"태그 '{tag_key}: {tag_value}'가 실행 {run_id}에 추가되었습니다."

    except Exception as e:
        return f"오류 발생: {str(e)}"

def log_command_interface(df, run_selection, command):
    """실행에 명령어 로깅"""
    if df is None or df.empty or not run_selection or not command:
        return "실행과 명령어를 입력해주세요."

    try:
        run_idx = int(run_selection)
        run_id = df.iloc[run_idx]['run_id']

        tracker.log_command(run_id, command)
        return f"명령어가 실행 {run_id}에 로깅되었습니다."

    except Exception as e:
        return f"오류 발생: {str(e)}"

# Gradio 인터페이스 생성
with gr.Blocks(title="MLflow 실험 추적 대시보드") as app:
    gr.Markdown("# MLflow 실험 추적 및 분석 대시보드")

    # 상태 저장용 변수
    experiment_data = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 실험 선택 및 필터링")
            refresh_btn = gr.Button("실험 목록 새로고침")
            experiment_dropdown = gr.Dropdown(label="실험 선택", choices=[])
            tag_filter = gr.Textbox(label="태그 필터 (쉼표로 구분)", placeholder="tag1,tag2,tag3")
            load_btn = gr.Button("데이터 로드", variant="primary")
            load_status = gr.Textbox(label="상태", interactive=False)

        with gr.Column(scale=3):
            gr.Markdown("## 실행 목록")
            runs_table = gr.Dataframe(label="실행 데이터", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 손실 값 비교")
            loss_columns = gr.Textbox(label="표시할 손실 컬럼 (쉼표로 구분)",
                                    placeholder="metric_loss,metric_val_loss")
            loss_plot = gr.Plot(label="손실 비교 차트")

        with gr.Column():
            gr.Markdown("## 시간별 메트릭")
            time_plot = gr.Plot(label="시간별 메트릭 차트")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 실행 세부 정보")
            run_selector = gr.Number(label="실행 인덱스 (테이블의 행 번호)", value=0, precision=0)
            run_details = gr.Markdown(label="실행 세부 정보")

        with gr.Column():
            gr.Markdown("## 태그 및 명령어 관리")
            with gr.Group():
                gr.Markdown("### 태그 추가")
                tag_key = gr.Textbox(label="태그 키")
                tag_value = gr.Textbox(label="태그 값")
                add_tag_btn = gr.Button("태그 추가")
                tag_status = gr.Textbox(label="태그 추가 상태", interactive=False)

            with gr.Group():
                gr.Markdown("### 명령어 로깅")
                command_input = gr.Textbox(label="실행 명령어", placeholder="python train.py --lr 0.001")
                log_cmd_btn = gr.Button("명령어 로깅")
                cmd_status = gr.Textbox(label="명령어 로깅 상태", interactive=False)

    # 이벤트 핸들러
    refresh_btn.click(refresh_experiments, outputs=experiment_dropdown)

    load_btn.click(
        load_experiment_data,
        inputs=[experiment_dropdown, tag_filter],
        outputs=[experiment_data, load_status]
    ).then(
        lambda df: df if df is not None else gr.update(),
        inputs=experiment_data,
        outputs=runs_table
    )

    # 플롯 업데이트
    loss_columns.change(
        plot_loss_comparison,
        inputs=[experiment_data, loss_columns],
        outputs=loss_plot
    )

    experiment_data.change(
        plot_metrics_over_time,
        inputs=experiment_data,
        outputs=time_plot
    )

    # 실행 세부 정보 업데이트
    run_selector.change(
        get_run_details,
        inputs=[experiment_data, run_selector],
        outputs=run_details
    )

    # 태그 추가
    add_tag_btn.click(
        add_tag_interface,
        inputs=[experiment_data, run_selector, tag_key, tag_value],
        outputs=tag_status
    )

    # 명령어 로깅
    log_cmd_btn.click(
        log_command_interface,
        inputs=[experiment_data, run_selector, command_input],
        outputs=cmd_status
    )

if __name__ == "__main__":
    app.launch(share=True)