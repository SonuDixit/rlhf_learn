import streamlit as st  
import pandas as pd  
import matplotlib.pyplot as plt  
  
# Function to read and parse the log file  
def parse_log_file(log_file_path):  
    data = {'Episode': [], 'Train Loss': [], 'Avg Val Reward': []}  
    episode_train_loss = {}  
    episode_val_reward = {}  
  
    with open(log_file_path, 'r') as file:  
        for line in file:  
            if 'episode:' in line:  
                st_ind = line.index('episode')
                line = line[st_ind:]
                ep_num, val = line.split(',')  
                episode_num = float(ep_num.split(':')[1].strip())  
                  
                if 'train_loss' in val:  
                    train_loss = float(val.split(':')[1].strip())  
                    episode_train_loss[episode_num] = train_loss  
                elif 'avg_val_reward' in val:  
                    avg_val_reward = float(val.split(':')[1].strip())  
                    episode_val_reward[episode_num] = avg_val_reward    
    
    train_df = pd.DataFrame(list(episode_train_loss.items()), columns=['Episode', 'train_loss'])
    val_df = pd.DataFrame(list(episode_val_reward.items()), columns=['Episode', 'avg_val_reward'])
    return train_df, val_df
  
# Streamlit UI  
st.title('Training Metrics Visualization')  
  
log_file_path = st.text_input('Enter the path to your log file', 'train.log')  

# Add a submit button
if st.button('Submit'):
    if log_file_path:  
        train_df, val_df = parse_log_file(log_file_path)  
          
        if not train_df.empty:  
            # Plotting Training Loss  
            if 'train_loss' in train_df.columns:  
                st.subheader('Training Loss per Episode')  
                fig, ax = plt.subplots()  
                ax.plot(train_df['Episode'], train_df['train_loss'], linestyle='-', label='Train_loss')  
                ax.set_xlabel('Episode')  
                ax.set_ylabel('Training Loss')  
                st.pyplot(fig)  
              
            # Plotting Average Validation Reward  
            if 'avg_val_reward' in val_df.columns:  
                st.subheader('Average Validation Reward per Episode')  
                fig, ax = plt.subplots()  
                ax.plot(val_df['Episode'], val_df['avg_val_reward'], linestyle='-', label='avg_val_reward')  
                ax.set_xlabel('Episode')  
                ax.set_ylabel('Average Validation Reward')  
                st.pyplot(fig)  
        else:  
            st.write("No data found in the log file.")  
