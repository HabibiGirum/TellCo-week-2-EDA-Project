# week-2 project TellCo


![Build Status](https://github.com/HabibiGirum/Stock_Market_Insights_From_Financial_news/actions/workflows/unittests.yml/badge.svg)


## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- Git installed
- A virtual environment tool (e.g., `venv`)

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/HabibiGirum/TellCo-week-2-EDA-Project.git
   cd TellCo-week-2-EDA-Project
   ```

2. **first Create and Activate Virtual Environment**

   ```bash
  
   python -m venv env # python 2 version
   python3 -m venv env # python3 version
   source venv/bin/activate  # for macOS and linux platforms
   venv\Scripts\activate # for Windows
   ```

3. **Install Dependencies**

   ```bash
   pip3 install -r requirements.txt
   ```

Here is a structured README format for the dataset used in the project:

---



## **Dataset Overview**
This dataset contains session-level data for customers of a telecommunication service provider. It is designed to analyze user engagement and activity patterns across various applications and services. The data is crucial for assessing Quality of Service (QoS) and optimizing resource allocation based on user behavior.

---

## **Column Descriptions**

| **Column Name**              | **Description**                                                                 |
|-------------------------------|---------------------------------------------------------------------------------|
| `MSISDN/Number`                     | Unique identifier for the customer (Mobile Subscriber Integrated Services Digital Network). |
| `Dur. (ms)`                  | Total duration of user sessions in milliseconds.                                |
| `Total DL (Bytes)`           | Total download traffic in bytes during the session.                             |
| `Total UL (Bytes)`           | Total upload traffic in bytes during the session.                               |
| `bearer id`                  | Number of sessions for the customer during the dataset period.                  |
| `Social Media DL (Bytes)`    | Download traffic for social media apps (e.g., Facebook, Instagram).             |
| `Social Media UL (Bytes)`    | Upload traffic for social media apps (e.g., Facebook, Instagram).               |
| `Youtube DL (Bytes)`         | Download traffic for YouTube application.                                       |
| `Youtube UL (Bytes)`         | Upload traffic for YouTube application.                                         |
| `Netflix DL (Bytes)`         | Download traffic for Netflix application.                                       |
| `Netflix UL (Bytes)`         | Upload traffic for Netflix application.                                         |
| `Google DL (Bytes)`          | Download traffic for Google services (e.g., Search, Maps).                      |
| `Google UL (Bytes)`          | Upload traffic for Google services (e.g., Search, Maps).                        |
| `Email DL (Bytes)`           | Download traffic for email services.                                            |
| `Email UL (Bytes)`           | Upload traffic for email services.                                              |
| `Gaming DL (Bytes)`          | Download traffic for gaming applications.                                       |
| `Gaming UL (Bytes)`          | Upload traffic for gaming applications.                                         |

---



### References

Python Libraries

```pandas``` for data manipulation
```matplotlib``` and ```seaborn``` for data visualization


## Deploy :
[click me](https://tellco-week-2-eda-project-zbwuw4qlgrrrnmjmj2d9fp.streamlit.app/)


## Author  
GitHub: [HabibiGirum](https://github.com/HabibiGirum)

Email:  habtamugirum478@gmail.com

