import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import imageio
import numpy as np
import plotly.io as pio
import io
from PIL import Image

class TimeSeriesVisualizer:
    def __init__(self):
        pass

    def visualize_2d(self, df: pl.DataFrame, 
        x_column="dim0", y_column="dim1", label_column="label", colors=["blue", "red"], t_column="t",
        x_axis_title="dim0", y_axis_title="dim1", x_range=[-40, 40], y_range=[-40, 40], dtick=10):

        unique_labels = list(set(df[label_column].to_list()))
        sorted_labels = sorted(unique_labels)

        pd_df = df.to_pandas()

        fig = px.scatter(
            data_frame=pd_df,  # 使用するデータフレーム
            x=x_column,  # 横軸の値に設定するデータフレームの列名
            y=y_column,  # 縦軸に設定するデータフレームの列名
            color=label_column,  # 色を区別する列名
            hover_data=pd_df,
            color_continuous_scale=colors,
            animation_frame=t_column, 
            category_orders={label_column: sorted_labels},
        )

        fig.update_layout(
            width=500, 
            height=500,
            xaxis = dict(title=x_axis_title, range = x_range, dtick=dtick),   
            yaxis = dict(title=y_axis_title, range = y_range, dtick=dtick, scaleanchor='x')
        )
        fig.update_layout(coloraxis_showscale=False)

        return fig

    
    def visualize_3d(
        self, df: pl.DataFrame, x_column="dim0", y_column="dim1", z_column="dim2", t_column="t", label_column="label",
        x_axis_title="x", y_axis_title="y", z_axis_title="z", x_range=[-4, 4], y_range=[-4, 4], z_range=[-4, 4], dtick=10
        ):

        # unique_labels = list(set(df[label_column].to_list()))
        # sorted_labels = sorted(unique_labels)
        
        pd_df = df.to_pandas()
        # 3次元
        fig = px.scatter_3d(
            data_frame=pd_df,  # 使用するデータフレーム
            x=x_column,  # x軸の値に設定するデータフレームの列名
            y=y_column,  # y軸に設定するデータフレームの列名
            z=z_column,  # z軸に設定するデータフレームの列名
            color=label_column,  # 色を区別する列名
            animation_frame=t_column,  # アニメーションのフレームとなるデータフレームの列名
            width=500,
            height=500,
            size=[0.5 for i in range(pd_df.shape[0])],
            size_max=5,
            opacity=1,
            range_x=x_range,
            range_y=y_range,
            range_z=z_range,
            labels={
                x_column: x_axis_title,
                y_column: y_axis_title,
                z_column: z_axis_title
            },
            color_continuous_scale=["red", "blue"],
        )

        # レイアウトの設定
        fig.update_layout(
            scene=dict(
                xaxis=dict(title=x_axis_title, range=x_range, dtick=dtick),
                yaxis=dict(title=y_axis_title, range=y_range, dtick=dtick),
                zaxis=dict(title=z_axis_title, range=z_range, dtick=dtick),
            ),
            coloraxis_showscale=False
        )
                    
        return fig

    def save_fig_as_gif(self, fig, filename, x_range, y_range, z_range=None, dtick=1, colorscale=px.colors.qualitative.Plotly, fps=2):
        
        is_3d = False if z_range is None else True
        if is_3d:
            scene = go.layout.Scene(
                xaxis=dict(range=x_range, dtick=dtick),
                yaxis=dict(range=y_range, dtick=dtick),
                zaxis=dict(range=z_range, dtick=dtick),
            )
        else:
            xaxis=dict(range=x_range, dtick=dtick)
            yaxis=dict(range=y_range, dtick=dtick, scaleanchor='x', scaleratio = 1.0)
        
        with imageio.get_writer(f"{filename}.gif", mode="I", fps=fps) as writer:
            for frame in fig.frames:
                if is_3d:
                    frame.layout.scene = scene
                    frame.layout.scene.aspectratio=dict(x=1, y=1, z=1)
                else:
                    frame.layout.xaxis = xaxis
                    frame.layout.yaxis = yaxis
                    if "annotations" in fig.layout:
                        frame.layout.annotations = fig.layout.annotations
                        
                    if "shapes" in fig.layout:
                        frame.layout.shapes = fig.layout.shapes
                        
                frame.layout.coloraxis.showscale=False
                frame.layout.colorscale = dict(sequential=colorscale)
                img_bytes = pio.to_image(frame, format="png", width=500, height=500)
                img_np = np.array(Image.open(io.BytesIO(img_bytes)))
                writer.append_data(img_np)