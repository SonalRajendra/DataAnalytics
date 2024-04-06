import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_process import Interpolation, OutlierDetection, Plotter, Smoothing


def upload_file():  # Function to display file uploader and process uploaded file
    upload_bt_help_text = """
                            \nNote: 
                            \n*Each column in uploaded file should have only one header. 
                            \n*Column names should not be in brackets (). 
                            \n*Ensure Units if required.
                            """
    uploaded_file = st.file_uploader(
        "Upload file", type=["csv", "xlsx", "xls"], help=upload_bt_help_text
    )
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return
        return df
    return None
    # Function to select column from the dataset


def select_column(columns_list):
    selected_columns = st.multiselect("Select column(s)", columns_list)
    return selected_columns


def save_dataframe(cleaned_data, filename, filetype):
    df = pd.DataFrame(cleaned_data)  # Create a DataFrame from the cleaned data
    if (
        filetype.lower() == "csv"
    ):  # Determine the file extension based on the selected filetype
        return (
            df.to_csv(index=False).encode("utf-8"),
            f"{filename}.csv",
            "text/csv",
        )  # Save as CSV
    elif filetype.lower() == "excel":
        # Save as Excel
        return (
            df.to_excel(index=False).encode("utf-8"),
            f"{filename}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    elif filetype.lower() == "text":
        # Save as Text
        return (
            df.to_csv(sep="\t", index=False).encode("utf-8"),
            f"{filename}.txt",
            "text/plain",
        )
    else:
        return None, None, None


def plot_scatter_with_outliers(
    original_df, cleaned_df, column_names
):  # Create a scatter plot for each column
    for column in column_names:
        fig0, aj = plt.subplots()  # Creating a new figure and axes
        # Plotting original data
        aj.scatter(
            original_df.index, original_df[column], color="blue", label="Original Data"
        )
        outliers = cleaned_df[
            column
        ].isna()  # Plotting outliers, Highlight outliers in red
        aj.scatter(
            original_df.index[outliers],
            original_df[column][outliers],
            color="red",
            label="Outliers",
        )
        aj.set_xlabel("Index")  # Setting x-axis label
        aj.set_ylabel(column)  # Setting y-axis label
        aj.set_title("Original Data with Outliers Highlighted")  # Setting plot title
        aj.legend()  # Adding legend
        return fig0  # Returning the figure


def plot_scatter_multi_with_outliers(original_df, cleaned_df, column):
    fig0, aj = plt.subplots()  # Creating a new figure and axes
    # Plotting original data
    aj.scatter(
        original_df.index, original_df[column], color="blue", label="Original Data"
    )
    outliers = cleaned_df[column].isna()  # Highlight outliers in red
    # Plotting outliers
    aj.scatter(
        original_df.index[outliers],
        original_df[column][outliers],
        color="red",
        label="Outliers",
    )
    aj.set_xlabel("Index")  # Setting x-axis label
    aj.set_ylabel(column)  # Setting y-axis label
    aj.set_title("Original Data with Outliers Highlighted")  # Setting plot title
    aj.legend()  # Adding legend
    return fig0  # Returning the figure


def display_raw(
    df, selected_x_column, selected_y_column, chart_type
):  # Function to display raw data
    if selected_x_column == "index":
        x_axis = df.index
        if chart_type == "Line Chart":
            raw_fig, ax = plt.subplots()  # Create a line chart
            ax.plot(x_axis, df[selected_y_column], color="red")
            ax.set_xlabel(selected_x_column)
            ax.set_ylabel(selected_y_column)
            return raw_fig
        elif chart_type == "Bar Chart":
            raw_fig, ax = plt.subplots()  # Create a bar chart
            ax.bar(x_axis, df[selected_y_column], color="red")
            ax.set_xlabel(selected_x_column)
            ax.set_ylabel(selected_y_column)
            return raw_fig
    else:
        x_axis = df[selected_x_column]
        if chart_type == "Line Chart":
            raw_fig, ax = plt.subplots()  # Create a line chart
            ax.plot(x_axis, df[selected_y_column], color="red")
            ax.set_xlabel(selected_x_column)
            ax.set_ylabel(selected_y_column)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            return raw_fig
        elif chart_type == "Bar Chart":
            raw_fig, ax = plt.subplots()  # Create a bar chart
            ax.bar(x_axis, df[selected_y_column], color="red")
            ax.set_xlabel(selected_x_column)
            ax.set_ylabel(selected_y_column)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            return raw_fig
            # Function to display the result data


def display_processed(
    Interpolated_data,
    cleaned_data,
    selected_x_column_f,
    selected_y_column_f,
    chart_type_f,
    num_points,
):
    if selected_x_column_f == "index":
        x_axis = cleaned_data.index[:num_points]
        if isinstance(
            x_axis, pd.DatetimeIndex
        ):  # Check if x_axis is already datetime index
            x_axis_date_strings = x_axis.strftime(
                "%Y-%m-%d"
            )  # Convert directly to string
        else:
            x_axis_date_strings = x_axis.astype(
                str
            )  # Convert other index types to string
    else:
        x_axis = cleaned_data[selected_x_column_f][:num_points]
        x_axis_datetime = pd.to_datetime(x_axis, errors="coerce")
        x_axis_date_strings = x_axis_datetime.dt.strftime("%Y-%m-%d")

    x_axis_ticks = [0, len(x_axis) // 2, len(x_axis) - 1]
    x_axis_labels = x_axis_date_strings[x_axis_ticks]

    if chart_type_f == "Line Chart":
        fig, ax = plt.subplots()  # Create a line chart
        ax.plot(
            x_axis,
            cleaned_data[selected_y_column_f][:num_points],
            color="red",
            label="Smooth Data",
        )
        ax.plot(
            x_axis,
            Interpolated_data[selected_y_column_f][:num_points],
            color="blue",
            label="Interpolated Data",
        )
        ax.set_xlabel(selected_x_column_f)
        ax.set_ylabel(selected_y_column_f)
        ax.set_xticks(x_axis_ticks)
        ax.set_xticklabels(x_axis_labels)
        ax.legend()
        return fig

    elif chart_type_f == "Bar Chart":
        fig, ax = plt.subplots()  # Create a bar chart
        ax.bar(
            x_axis,
            cleaned_data[selected_y_column_f][:num_points],
            color="red",
            label="Smooth Data",
        )
        ax.bar(
            x_axis,
            Interpolated_data[selected_y_column_f][:num_points],
            color="blue",
            label="Interpolated Data",
        )
        ax.set_xlabel(selected_x_column_f)
        ax.set_ylabel(selected_y_column_f)
        ax.set_xticks(x_axis_ticks)
        ax.set_xticklabels(x_axis_labels)
        ax.legend()
        return fig


def matlab_to_datetime(
    matlab_time,
):  # Function to convert MATLAB serial date number to Python datetime object
    # MATLAB serial date number starts from January 1, 0000, and Python's datetime starts from January 1, 0001
    # Need to adjust the offset by subtracting the MATLAB epoch offset
    python_epoch = datetime.datetime(1, 1, 1)
    matlab_epoch = (
        datetime.datetime.fromordinal(1)
        + datetime.timedelta(days=366)
        - datetime.timedelta(days=1)
    )
    offset = matlab_epoch - python_epoch
    # Convert MATLAB time to Python datetime
    python_datetime = (
        datetime.datetime.fromordinal(int(matlab_time))
        + datetime.timedelta(days=matlab_time % 1)
        - offset
    )
    return python_datetime

    # Function to rename selected columns with a prefix


def rename_columns_with_prefix(df, selected_columns):
    renamed_df = df.copy()  # Make a copy of the DataFrame
    prefix = "p_"
    for column in selected_columns:
        renamed_df.rename(columns={column: prefix + column}, inplace=True)
    return renamed_df

    # Function for the button state


def set_state(i):
    st.session_state.stage = i

    # Main function to run the Streamlit app


def main():
    st.title("Data Filteration and Linearization Tool")
    st.divider()
    df = upload_file()
    if df is not None:
        # Setting state for the buttons used
        if "stage" not in st.session_state:
            st.session_state.stage = 0
        with st.expander("See Uploaded  Data"):
            st.write("User Data:", df)
            # Convert the 'Datum' column to Python datetime
            column_names = df.columns
            for column_name in column_names:
                if "Datum" in column_name or "Matlab_Time" in column_name:
                    converted_dt = df["Datum"].apply(matlab_to_datetime)
                    df["Datum"] = converted_dt
                    df.rename(columns={"Datum": "Processed Date Time"}, inplace=True)
                    st.write("User Data with updated DateTime:", df)
                    # Button to display result as a plot
            chart_types = ["Line Chart", "Bar Chart"]
            # Selectbox for choosing chart type
            chart_type = st.selectbox("Select Chart Type", chart_types)

            columns_list = df.columns.tolist()
            selected_x_column = st.selectbox(
                "Select x-axis Column", ["index"] + columns_list
            )
            selected_y_column = st.selectbox("Select y-axis Column", columns_list)

            if st.button("Display Result"):
                result_chart = display_raw(
                    df, selected_x_column, selected_y_column, chart_type
                )
                if result_chart is not None:
                    st.pyplot(result_chart)
                else:
                    st.error("Invalid chart type selected.")

        columns_list = (
            df.columns.tolist()
        )  # Read column and select from the uploaded file
        selected_columns = select_column(columns_list)

        with st.expander("See Overview"):
            st.subheader("Methods Overview:")

            if st.checkbox("Outlier Detection", key="chk_outlier"):
                outlier_text = """
                                        \n1. Z-score Method: Identifies outliers by calculating the Z-scores for the dataset. The Z-score measures 
                                        how many standard deviations a data point is from the mean. By setting a threshold, specifies the 
                                        number of standard deviations away from the mean that constitutes an outlier. This threshold defines what is 
                                        considered an outlier in your dataset.
                                        \n2. Interquartile Range (IQR) Method: Detects outliers based on the Interquartile Range, 
                                        identifying data points falling below the first quartile minus a threshold 
                                        or above the third quartile plus a threshold. 
                                        \n3. Isolation Forest Method: Utilizes the Isolation Forest algorithm to isolate outliers
                                        by randomly selecting features and splitting data points into smaller groups.
                                        """
                st.markdown(outlier_text)
                st.link_button(
                    "Know More",
                    "https://en.wikipedia.org/wiki/Outlier",
                    help="Click here to know more aboout the above topics.",
                )
                st.divider()

            if st.checkbox("Data Interpolation", key="chk_interpol"):
                interpol_text = """
                                        \n1. Liner Interpolation: linear interpolation is a method of curve fitting using linear polynomials 
                                        to construct new data points within the range of a discrete set of known data points. 
                                        \n2. Quadratic Interpolation: qudratic interpolation is a method of curve fitting using second degree 
                                        polynomials to construct new data points within the range of a discrete set of known data points. 
                                        \n3. Cubic Interpolation: In cubic spline interpolation the interpolating function is a set of piecewise 
                                        cubic functions between each of the data points. 
                                        """
                st.markdown(interpol_text)
                st.link_button(
                    "Know More",
                    "https://nm.mathforcollege.com/NumericalMethodsTextbookUnabridged/chapter-05.05-spline-method-of-interpolation.html",
                    help="Click here to know more aboout the above topics.",
                )
                st.divider()

            if st.checkbox(" Data Smoothing", key="chk_smooth"):
                smooth_text = """
                                        \n1. Moving average: This method entails selecting a window of data points, computing their average, shifting the window by one point, 
                                        and repeating the process. This iterative approach generates smoothed data points. 
                                        \n2. Savitzky-Golay filter: A Savitzky–Golay filter smooths digital data points without distorting the signal tendency. It fits 
                                        adjacent data subsets with a low-degree polynomial using linear least squares convolution."""
                st.markdown(smooth_text)
                st.link_button(
                    "Know More",
                    "https://pieriantraining.com/python-smoothing-data-a-comprehensive-guide/",
                    help="Click here to know more aboout the above topics.",
                )
                st.divider()
        text1 = """Choose below methods to perform on the selected column(s)"""
        st.text(text1)

        toggle_outlier = st.toggle("Outlier Detection")  # Toggle button for outlier
        if toggle_outlier:  # Check the state of the toggle button
            outliers_option = st.radio(
                "Choose an outlier method",
                ["Z-Score Method", "IQR Method", "Isolation Forest"],
            )
            if outliers_option in ["Z-Score Method", "IQR Method"]:
                threshold = st.slider(
                    label="Threshold for outlier detection",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                )
                st.info(
                    "ℹ️ Threshold determines the distance from the mean at which data points are considered outliers, influencing the sensitivity of outlier detection."
                )

            elif outliers_option in ["Isolation Forest"]:
                contamination_rate = st.slider(
                    label="Contamination rate for outlier detection",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.05,
                    step=0.01,
                )
                st.info(
                    "ℹ️ The contamination rate represents the proportion of anomalies expected in the dataset and influences the sensitivity of outlier detection."
                )
            st.divider()

        toggle_interpol = st.toggle(
            "Data interpolation"
        )  # Toggle button for interpolation
        if toggle_interpol:  # Check the state of the toggle button
            interpolations_option = st.radio(
                "Choose an Interpolation method", ["linear", "quadratic", "cubic"]
            )
            st.write(
                f"Interpolation method selected: {interpolations_option}"
            )  # Interpolations_option in ["linear", "quadratic","cubic"]
            st.divider()

        toggle_smoothing = st.toggle("Data Smoothing")  # Toggle button for smoothing
        if toggle_smoothing:  # Check the state of the toggle button
            st.subheader("Smoothing Options")
            smoothing_option = st.radio(
                "Choose a smoothing method", ["Moving Average", "Savitzky-Golay filter"]
            )

            if smoothing_option == "Moving Average":
                st.subheader("Smoothing Parameters")

                st.write(
                    "Window Size for Moving Average:"
                )  # Help icon and details for moving average window size
                st.info(
                    "ℹ️ Window size controls the number of data points to consider for each moving average calculation."
                )
                filter_length = st.slider(
                    label="Window Size", min_value=1, max_value=100, value=10, step=1
                )

            elif smoothing_option == "Savitzky-Golay filter":
                st.subheader("Smoothing Parameters")

                st.write(
                    "Filter Length for Smoothing:"
                )  # Help icon and details for filter length
                st.info("""ℹ️ Filter length controls the size of the window used for smoothing. 
                        Choose it according to the characteristic length scale of your data.""")
                filter_length1 = st.slider(
                    label="Filter Length",
                    min_value=0,
                    max_value=1000,
                    value=500,
                    step=25,
                )

                st.write(
                    "Order of Smoothing:"
                )  # Help icon and details for order length
                st.info("""ℹ️ The order of smoothing refers to the polynomial order used in 
                        the Savitzky-Golay filter. It must be less than the filter length.""")
                order_length = st.slider(
                    label="Order of Smoothing",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1,
                )
            st.divider()

        left, center, right = st.columns(3)
        with left:
            if st.session_state.stage == 0:  # Check the state of the preprocess button
                st.button(
                    "Preprocess Data",
                    help="Click here to process the selected column for preprocessing",
                    on_click=set_state,
                    args=[1],
                )
        with right:
            st.button(
                "Reset",
                help="Click here to reset the process",  # Button to reset the output
                on_click=set_state,
                args=[0],
            )

        if (
            toggle_outlier == True and st.session_state.stage >= 1
        ):  # Logic to perform preprocessing (Outlier removal)
            # Initialize class objects
            outlier_detection = OutlierDetection(df)
            outlier_plots = Plotter(df)

            # Perform preprocessing steps
            if outliers_option in ["Z-Score Method"]:
                # Detect outliers using z-score
                outlier_result = outlier_detection.detect_outliers_zscore_drop(
                    selected_columns, threshold
                )
                st.write("Cleaned Data:", outlier_result)
                outlier_result.to_csv("Cleaned_data.csv", index=False)
                cleaned_data = outlier_result

            elif outliers_option in ["IQR Method"]:
                outlier_result = outlier_detection.detect_outliers_iqr_drop(
                    selected_columns, threshold
                )
                st.write("Cleaned Data:", outlier_result)
                outlier_result.to_csv("Cleaned_data.csv", index=False)
                cleaned_data = outlier_result

            elif outliers_option in ["Isolation Forest"]:
                outlier_result = outlier_detection.detect_outliers_isof_drop(
                    selected_columns, contamination_rate
                )
                st.write("Cleaned Data:", outlier_result)
                outlier_result.to_csv("Cleaned_data.csv", index=False)
                cleaned_data = outlier_result

            if len(selected_columns) == 1:
                outlier_plots = plot_scatter_with_outliers(
                    df, cleaned_data, selected_columns
                )
                st.pyplot(outlier_plots)
            elif len(selected_columns) > 1:
                with st.expander("See Outlier Result Data"):
                    y_col = st.selectbox(
                        "Column Name",
                        selected_columns,
                        help="Select the column name from drop down to plot.",
                    )
                    outlier_plots = plot_scatter_multi_with_outliers(
                        df, cleaned_data, y_col
                    )
                    st.pyplot(outlier_plots)

                    # Perform interpolation
        if (
            toggle_outlier == True
            and toggle_interpol == True
            and st.session_state.stage >= 1
        ):
            Cleaned_data = pd.read_csv("Cleaned_data.csv")  # Read Outlier removed data
            interpolation = Interpolation()  # Intialise class object
            interpolation_plot = Plotter(Cleaned_data)  # Intialise class object

            interpolated_df = interpolation.interpolation(
                Cleaned_data, selected_columns, method=interpolations_option
            )
            st.write("Interpolated Result:", interpolated_df)
            cleaned_data = interpolated_df
            interpolated_df.to_csv("Interpolated_data.csv", index=False)

            if len(selected_columns) == 1:
                interpolation_plot = interpolation_plot.plot_interpolated(
                    interpolated_df, selected_columns
                )
                st.pyplot(interpolation_plot)
            elif len(selected_columns) > 1:
                with st.expander("See Interpolated Result Data"):
                    y_col_interpol = st.selectbox(
                        "Name",
                        selected_columns,
                        help="Select the column name from drop down to plot.",
                    )
                    interpolation_plot = interpolation_plot.plot_interpolated(
                        interpolated_df, y_col_interpol
                    )
                    st.pyplot(interpolation_plot)

                    # Perform smoothing
        if toggle_smoothing == True and st.session_state.stage >= 1:
            try:
                Interpolated_data = pd.read_csv("Interpolated_data.csv")
                if Interpolated_data.empty:
                    raise FileNotFoundError
            except FileNotFoundError:
                Interpolated_data = (
                    df.copy()
                )  # If the file is empty or not present, use the original DataFrame df

            smoothing_detection = Smoothing(Interpolated_data)  # Intialise class object

            if smoothing_option in ["Moving Average"]:
                smoothing_result = smoothing_detection.moving_average(
                    selected_columns, filter_length
                )
                st.write("Moving Average Result", smoothing_result)
                cleaned_data = smoothing_result

            elif smoothing_option in ["Savitzky-Golay filter"]:
                smoothing_result = smoothing_detection.savitzky_golay(
                    selected_columns, filter_length1, order_length
                )
                st.write("Savitzky-Golay result", smoothing_result)
                cleaned_data = smoothing_result

            with st.expander("See Smooth Result Data"):  # Show result
                chart_types_f = ["Line Chart", "Bar Chart"]  # Available chart types
                chart_type_f = st.selectbox(
                    "Select Chart  Type", chart_types_f
                )  # Selectbox for choosing chart type

                columns_list_f = df.columns.tolist()
                selected_x_column_f = st.selectbox(
                    "Select X-axis Column", ["index"] + columns_list_f
                )
                selected_y_column_f = st.selectbox(
                    "Select Y-axis Column", columns_list_f
                )

                num_points = st.number_input(
                    "Select Number of Points to Plot",
                    min_value=0,
                    max_value=2000,
                    value=1000,
                    step=10,
                )

                if st.button("Display  Result"):
                    Interpolated_data = pd.read_csv("Interpolated_data.csv")
                    result_chart = display_processed(
                        Interpolated_data,
                        cleaned_data,
                        selected_x_column_f,
                        selected_y_column_f,
                        chart_type_f,
                        num_points,
                    )
                    if result_chart is not None:
                        st.pyplot(result_chart)
                    else:
                        st.error("Invalid chart type selected.")

        if st.session_state.stage >= 1:  # Download the result
            filename = st.text_input(
                "File Name", on_change=set_state, args=[2]
            )  # Specify the file name
            filetype = st.selectbox(
                "File Type", options=["CSV", "Excel", "Text"]
            )  # Select file type

            if st.session_state.stage >= 2:  # Rename the processed columns
                cleaned_data = rename_columns_with_prefix(
                    cleaned_data, selected_columns
                )
                cleaned_data = pd.DataFrame(cleaned_data)
                if filename and filetype:
                    file_content, file_name_with_extension, mime_type = save_dataframe(
                        cleaned_data, filename, filetype
                    )
                    st.download_button(
                        "Download",  # save the file as per the filename and type
                        data=file_content,
                        file_name=file_name_with_extension,
                        mime=mime_type,
                        on_click=set_state,
                        args=[3],
                        help="Click here to save the cleaned data as an excel file.",
                    )
            else:
                set_state(2)

        if st.session_state.stage >= 3:
            st.write("The file is saved")
            st.button(
                "Start Over", on_click=set_state, args=[0]
            )  # Reset the buttong state so user can start over


if __name__ == "__main__":
    main()
