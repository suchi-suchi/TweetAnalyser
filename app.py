##  Frond end code file ##
import streamlit as st
import pandas as pd
import cleanData
import emotion_detection
import utils
import matplotlib.pyplot as plt
import model
# using streamlit library for UI Components 
st.title("Social Media Tweet Analyser")
uploaded_file = st.file_uploader("Choose a file")
# Getting trained model, tfidf object, features, target, SelectKbest objects from model.py file
naive_bayes, tfidf, train, y, X_new = model.getModel()
string_data = ""
unique_emotions = list(y.unique())
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = uploaded_file.getvalue().decode('utf-8')
    # Pre processing input text file by using cleanData.py file
    df = cleanData.getDataFrame(data)

    df = df[df["user"]!="group_notification"]
    user_list = df['user'].unique().tolist()
    user_dict = dict()
    for i in user_list:
        user_dict[i] = len(df[df["user"] == i])
    
    user_dict = dict(sorted(user_dict.items(), key=lambda item: item[1]))

    user_list = list()
    for i, j in user_dict.items():
        user_list.append(i)
    user_list = user_list[::-1]
    user_list.insert(0,"Group")
    # Preprocessing the group chat data (Stopwords removal, Stemming, Lemmatization etc.,)
    processed_df = emotion_detection.preprocess(df)
    final_df = processed_df[processed_df["updated_message"].isna()]
    final_df["emotion"] = pd.Series(["neutral" for i in range(len(final_df))])
    df_to_predict = processed_df[processed_df["updated_message"].notnull()]

    #Transforming to Tfidf
    tfidf_to_predict = tfidf.transform(df_to_predict["updated_message"])

    #Transforming to selected features extracted from SelectKBest Method
    new_predict_df = X_new.transform(tfidf_to_predict)

    #Predicting the Emotion of the messages
    predicted = naive_bayes.predict(new_predict_df)
    predicted = pd.DataFrame({"emotion": predicted})
    pred_df = pd.concat([df_to_predict, predicted], axis=1, join="inner")
    final_df = pd.concat([final_df, pred_df])
   
    #Check Box for Group or Individual user
    selected_user = st.selectbox("User List",user_list)
    if st.button("Visualize"):
        st.write("Computing")
        col1, col2 = st.columns(2)
        
        messages_length, words, media_length, num_links = utils.return_stats(df, selected_user)
        with col1:
            st.header("Total Messages")
            st.title(messages_length)
        with col2:
            st.header("Total Links Shared")
            st.title(num_links)
        

        #Wordcloud generation
        if selected_user != "Group":
            st.header("Most Common Words used by {}".format(selected_user))
        else:
            st.header("Most Common Words used in Group Overall")
        df_wc = utils.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        plt.axis("off")
        st.pyplot(fig)

        if selected_user != "Group":
            #Monthly Timeline generation
            st.title("Monthly Timeline")
            timeline = utils.monthly_timeline(selected_user,df)
            fig,ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'],color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


            st.title('Activity Map')
            col1,col2 = st.columns(2)

            with col1:

                #Most Busy Days sorted Generation
                st.header("Most busy day")
                busy_day = utils.week_activity_map(selected_user,df)
                fig,ax = plt.subplots()
                ax.bar(busy_day.index,busy_day.values,color='blue')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                #Most Busy Months sorted Generation
                st.header("Most busy month")
                busy_month = utils.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            #Emotion Visualization
            st.title("Emotion Visualization")
            
            user_custom_df = final_df[final_df["user"] == selected_user]
            emotion_dict = user_custom_df["emotion"].value_counts()
            fig6, ax = plt.subplots()
            for i in unique_emotions:
                if i not in emotion_dict:
                    emotion_dict[i] = 0
            ax.bar(emotion_dict.index, emotion_dict.values, color="orange")
            plt.xticks(rotation="vertical")
            st.pyplot(fig6)

        else:
            #Total Messages by every user
            st.header("Total messages by every user")
            result_messages_by_user = utils.get_messages_by_user(df)
            result_mostwords = utils.get_most_words_by_user(df)
            fig2, ax = plt.subplots()
            ax.set_xlabel("Words")
            ax.set_ylabel("Frequency")
            ax.bar(list(result_messages_by_user.keys()), list(result_messages_by_user.values()), color='black')
            plt.xticks(rotation='vertical', fontsize=4)
            st.pyplot(fig2)

            #Most Busy Users sorted Generation (Top 5)
            col1,col2 = st.columns(2)
            st.title('Most Busy Users')
            x,new_df = utils.most_busy_users(df)
            fig3, ax = plt.subplots()
            ax.bar(x.index, x.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig3)

            with col1:
                st.header("Most busy days")
                busy_day = utils.week_activity_map(selected_user,df)
                fig4,ax = plt.subplots()
                ax.bar(busy_day.index,busy_day.values,color='blue')
                plt.xticks(rotation='vertical')
                st.pyplot(fig4)

            with col2:
                st.header("Most busy months")
                busy_month = utils.month_activity_map(selected_user, df)
                fig5, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig5)

            fig6, ax = plt.subplots()
            emotion_dict = final_df["emotion"].value_counts()

            #Group Emotion Detection collectively
            for i in unique_emotions:
                if i not in emotion_dict:
                    emotion_dict[i] = 0
            ax.bar(emotion_dict.index, emotion_dict.values, color="orange")
            plt.xticks(rotation="vertical")
            st.pyplot(fig6)
