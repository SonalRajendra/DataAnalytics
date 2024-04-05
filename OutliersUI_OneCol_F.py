import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from OutliersMain_OneCol_F import OutlierDetection, Smoothing, Interpolation, TimeConverter, Plotter

# Function to display file uploader and process uploaded file
def upload_file():
    uploaded_file = st.file_uploader("Upload file", type=["csv","xlsx","xls"], 
                                     help="Note: Each column in uploaded file should have only one header. \nColumn names should not be in brackets (). \nEnsure Units if required.")
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return
        return df
    return None

# Function to select column from the dataset    
def select_column(columns_list):
    selected_columns = st.selectbox("Select a column", columns_list)
    return selected_columns

#Function to download the cleaned datset
def save_dataframe_as_csv(cleaned_data, filename):
    df = pd.DataFrame(cleaned_data)
    filename_concat= (f"{filename}.csv")
    df.to_csv(filename_concat, index=False)

#Function to display raw data
def display_raw(df, selected_x_column, selected_y_column, chart_type):
    if selected_x_column == "index":
        x_axis=df.index
    else:
        x_axis = df[selected_x_column]  
       
    if chart_type == "Line Chart":
    # Create a line chart
        raw_fig, ax = plt.subplots()
        ax.plot(x_axis, df[selected_y_column], color='red')
        ax.set_xlabel(selected_x_column)
        ax.set_ylabel(selected_y_column)
        return raw_fig
    
    elif chart_type == "Bar Chart":
    # Create a bar chart
        raw_fig, ax = plt.subplots()
        ax.bar(x_axis, df[selected_y_column], color='red')
        ax.set_xlabel(selected_x_column)
        ax.set_ylabel(selected_y_column)
        return raw_fig
         
    
#Function for the button state
def set_state(i):
    st.session_state.stage = i

# Main function to run the Streamlit app
def main():
    st.title("Data Filteration and Linearization Tool")
    st.divider()
    df = upload_file()
    if df is not None:
        if 'stage' not in st.session_state: #setting state for the buttons used
            st.session_state.stage = 0
        with st.expander("See Uploaded  Data"):
            st.write("User Data:", df)

            # Add button to display result as a plot
            chart_types = ["Line Chart", "Bar Chart"]  # Available chart types
            chart_type = st.selectbox("Select Chart Type", chart_types)  # Selectbox for choosing chart type

            columns_list = df.columns.tolist()
            selected_x_column = st.selectbox("Select x-axis Column", ["index"] + columns_list)
            selected_y_column = st.selectbox("Select y-axis Column", columns_list)

            if st.button("Display Result"):
                result_chart = display_raw(df,selected_x_column,selected_y_column, chart_type)
                if result_chart is not None:
                    st.pyplot(result_chart)
                else:
                    st.error("Invalid chart type selected.")

        # Read column and select from the uploaded file
        columns_list = df.columns.tolist()
        selected_column = select_column(columns_list)
        st.text("Choose below methods to perform on the selected column")

        left, right = st.columns(2)
        with left: 
            # Add a toggle for the toggle button
            toggle_outlier = st.toggle("Outlier Detection")
            # Check the state of the toggle button
            if toggle_outlier:
                outliers_option = st.radio("Choose an outlier method", ["Z-Score Method", "IQR Method", "Isolation Forest"])
                if outliers_option in ["Z-Score Method", "IQR Method"]:
                    threshold = st.slider(label = "Threshold for outlier detection" , min_value = 0.1 , max_value = 5.0 , value = 1.0 , step = 0.1)
                elif outliers_option in ["Isolation Forest"]:
                    contamination_rate = st.slider(label = "Contamination rate for outlier detection" , min_value = 0.0 , max_value = 0.5 , value = 0.05 , step = 0.01)
                st.divider()

        # Add a toggle for the toggle button
        toggle_interpol = st.toggle("Data interpolation")
        # Check the state of the toggle button
        if toggle_interpol:
            interpolations_option = st.radio("Choose an Interpolation method", ["linear", "quadratic", "cubic"])
            st.write(f"Interpolation method selected: {interpolations_option}") # interpolations_option in ["linear", "quadratic","cubic"]
            st.divider()

        # Add a toggle for the toggle button
        toggle_smoothing = st.toggle("Data Smoothing")
        # Check the state of the toggle button
        if toggle_smoothing:
            smoothing_option = st.radio("Choose an smoothing method ", ["Moving Average", "Savitzky-Golay filter"])
            if smoothing_option in ["Moving Average"]:
                    filter_length = st.slider(label = "Filter Length for smoothing" , min_value = 1 , max_value = 10 , value = 5 , step = 1)
            elif smoothing_option in ["Savitzky-Golay filter"]:
                    filter_length1 = st.slider(label = "Filter Length for smoothing" , min_value = 1.0 , max_value = 10.0 , value = 5.0 , step = 1.0)
                    order_length = st.slider(label = "Order of smoothing" , min_value = 1.0 , max_value = 10.0 , value = 5.0 , step = 1.0)
            st.divider()
        
        with right:
            if (st.checkbox("Check here for overview.", key='chk_Overview')):
                with st.sidebar:
                    st.title("Overview:")
                    if (st.checkbox("Outlier Detection", key='chk_outlier')):
                                outlier_text="1. Z-score Method: Identifies outliers by calculating the Z-scores for the dataset. \n2. Isolation Forest Method: Utilizes the Isolation Forest algorithm to isolate outliers by randomly selecting features and splitting data points into smaller groups. \n3. Interquartile Range (IQR) Method: Detects outliers based on the Interquartile Range, identifying data points falling below the first quartile minus a threshold or above the third quartile plus a threshold."
                                st.markdown(outlier_text)
                                st.link_button("Know More",
                                               "https://www.freecodecamp.org/news/how-to-detect-outliers-in-machine-learning/",
                                               help="Click here to know more about above topics.")
                                st.divider()
                    if (st.checkbox("Data Interpolation", key='chk_interpol')):
                                interpol_text="1. Liner Interpolation: linear interpolation is a method of curve fitting using linear polynomials to construct new data points within the range of a discrete set of known data points. \n2. Qudratic Interpolation: qudratic interpolation is a method of curve fitting using second degree polynomials to construct new data points within the range of a discrete set of known data points. \n3. Cubic Interpolation: In cubic spline interpolation the interpolating function is a set of piecewise cubic functions between each of the data points. "
                                st.markdown(interpol_text)
                                st.link_button("Know More", 
                                               "https://nm.mathforcollege.com/NumericalMethodsTextbookUnabridged/chapter-05.05-spline-method-of-interpolation.html",
                                               help="Click here to know more aboout the above topics.")  
                                st.divider()
                    if (st.checkbox(" Data Smoothing", key='chk_smooth')):
                                smooth_text="1. Moving average: This method entails selecting a window of data points, computing their average, shifting the window by one point, and repeating the process. This iterative approach generates smoothed data points. \n2. Savitzky-Golay filter: A Savitzky–Golay filter smooths digital data points without distorting the signal tendency. It fits adjacent data subsets with a low-degree polynomial using linear least squares convolution."
                                st.markdown(smooth_text)
                                st.link_button("Know More",
                                               "https://pieriantraining.com/python-smoothing-data-a-comprehensive-guide/",
                                               help="Click here to know more aboout the above topics.")
                                st.divider()

        left, center, right = st.columns(3)
        with left:
        # Check the state of the toggle button
            if st.session_state.stage == 0:
                st.button('Preprocess Data', help="Click here to process the selected column for preprocessing", on_click=set_state, args=[1])
        with right:
             st.button('Reset', help="Click here to reset the process", on_click=set_state, args=[0])

        # Add button to perform preprocessing (Outlier removal)
        if toggle_outlier == True and st.session_state.stage >= 1:
            outlier_detection = OutlierDetection(df) # Initialize class objects
            outlier_plots = Plotter(df) # Initialize class objects
            
            # Perform preprocessing steps
            if outliers_option in ["Z-Score Method"]:
                # Detect outliers using z-score
                outlier_result = outlier_detection.detect_outliers_zscore(selected_column, threshold)
                z_score_outliers_str = ', '.join(map(str, outlier_result))
                #st.write("Z-score outlier Index:", z_score_outliers_str)
                # Remove outliers from data
                cleaned_data = outlier_detection.remove_outliers(selected_column, outlier_result)
                cleaned_data_df = df.copy()
                cleaned_data_df[selected_column] = cleaned_data #column name = array
                with st.expander("See Cleaned Data"):
                    st.write("Cleaned Data:", cleaned_data)
                    st.write("Outliers removed User Dataframe:", cleaned_data_df)
                # Plot original data with outliers highlighted
                fig0 = outlier_plots.plot_data_with_outliers(selected_column, outlier_result)
                st.pyplot(fig0)
                cleaned_data_df.to_csv('Cleaned_data.csv', index=False)


            elif outliers_option in ["IQR Method"]:
                outlier_result = outlier_detection.detect_outliers_iqr(selected_column, threshold)
                iqr_outliers_str = ', '.join(map(str, outlier_result))
                #st.write("IQR outliers Index:", iqr_outliers_str)
                # Remove outliers from data
                cleaned_data = outlier_detection.remove_outliers(selected_column, outlier_result)
                cleaned_data_df = df.copy()
                cleaned_data_df[selected_column] = cleaned_data #column name = array
                with st.expander("See Cleaned Data"):
                    st.write("Cleaned Data:", cleaned_data)
                    st.write("Outliers removed User Dataframe:", cleaned_data_df)
                # Plot original data with outliers highlighted
                fig2 = outlier_plots.plot_data_with_outliers(selected_column, outlier_result)
                st.pyplot(fig2)
                cleaned_data_df.to_csv('Cleaned_data.csv', index=False)
               
            elif outliers_option in ["Isolation Forest"]:
                outlier_result = outlier_detection.detect_outliers_isof(selected_column, contamination_rate)
                isolationdata = pd.DataFrame({'dropIndex':outlier_result})
                cleaned_data = pd.merge(df, isolationdata, left_index=True, right_index=True)
                cleaned_data = cleaned_data[cleaned_data.dropIndex == 1]
                with st.expander("See Cleaned Data"):
                    st.write("Cleaned Data:", cleaned_data)
                outlier_result_plot = outlier_plots.plot_isof(df, selected_column, outlier_result)
                st.pyplot(outlier_result_plot) #show plot
                cleaned_data_df.to_csv('Cleaned_data.csv', index=False)

        # Add button to perform interpolation
        if toggle_outlier == True and toggle_interpol == True and st.session_state.stage >= 1:
            Cleaned_data = pd.read_csv('Cleaned_data.csv')
            #st.write("Interpolate input Data:", Cleaned_data)
            interpolation = Interpolation()
            interpolation_plot = Plotter(Cleaned_data)
            interpolated_df = interpolation.interpolation(Cleaned_data, selected_column, method=interpolations_option)
            interpolation_plot = interpolation_plot.plot_interpolated(interpolated_df ,selected_column)
            st.write("Interpolated Result:", interpolated_df)
            st.pyplot(interpolation_plot)
            cleaned_data = interpolated_df
            interpolated_df.to_csv('Interpolated_data.csv', index=False)


        # Add button to perform smoothing
        if toggle_outlier == True or toggle_interpol == True or toggle_smoothing == True and  st.session_state.stage >= 1:
            Interpolated_data = pd.read_csv('Interpolated_data.csv')
            smoothing_detection = Smoothing(Interpolated_data) #Intialise class object

            if smoothing_option in ["Moving Average"]:
                smoothing_result = smoothing_detection.moving_average(selected_column,filter_length)
                st.write("Moving Average Result", smoothing_result)
                cleaned_data = smoothing_result
                #cleaned_data = Interpolated_data.drop(smoothing_result)
            
            elif smoothing_option in ["Savitzky-Golay filter"]:
                smoothing_result = smoothing_detection.savitzky_golay(Interpolated_data,selected_column,filter_length1,order_length)
                st.write("Savitzky-Golay result", smoothing_result)
                cleaned_data = smoothing_result
                #cleaned_data = Interpolated_data.drop(smoothing_result)

        #show result and ask user to enter the file name
        if st.session_state.stage >= 1:
            with st.expander("See Result Data"):
                # Add button to display result as a plot
                chart_types_f = ["Line Chart", "Bar Chart"]  # Available chart types
                chart_type_f = st.selectbox("Select Chart  Type", chart_types_f)  # Selectbox for choosing chart type

                columns_list_f = df.columns.tolist()
                selected_x_column_f = st.selectbox("Select X-axis Column", ["index"] + columns_list_f)
                selected_y_column_f = st.selectbox("Select Y-axis Column", columns_list_f)

                if st.button("Display  Result"):
                    result_chart = display_raw(cleaned_data,selected_x_column_f,selected_y_column_f, chart_type_f)
                    if result_chart is not None:
                        st.pyplot(result_chart)
                    else:
                        st.error("Invalid chart type selected.")
            filename = st.text_input('File Name', on_change=set_state, args=[2])
            
            if st.session_state.stage >= 2:
                toggle_save = st.toggle('Save')
                if toggle_save:
                    save = st.button('Save' ,
                    help='Click here to save the cleaned data as an excel file.' ,
                    on_click=set_state, args=[3])
                    save_dataframe_as_csv(cleaned_data, filename)
            else:
                set_state(2)

        if st.session_state.stage >= 3:
            st.write('The file is saved')
            st.button('Start Over', on_click=set_state, args=[0])
    
            # Display output results
                #st.write("Z-score outliers:", outlier_result)
                #st.write("IQR outliers:", outlier_result)

if __name__ == "__main__":
    main()
