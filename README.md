<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
<!--     <img src="images/logo.png" alt="Logo" width="80" height="80"> -->
  </a>

  <h3 align="center">Intelligent Assistant for Smart Factory Power Management</h3>

  <p align="center">
  Development of an assistant that can predict and detect anomalies on energy consumption related variables, and is implemented in an architecture that allows for data gathering and monitoring from the factory floor 
  <br />
    <a href="https://github.com/zemaria2000/IntelligentAssistant"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/zemaria2000/IntelligentAssistant/blob/main/README.md">View Demo</a>
    ·
    <a href="https://github.com/zemaria2000/IntelligentAssistant/issues">Report Bug</a>
    ·
    <a href="https://github.com/zemaria2000/IntelligentAssistant/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
<!--         <li><a href="#base-requirements">Base requirements</a></li> -->
        <li><a href="#implemented-architecture">Implemented Architecture</a></li>
        <li><a href="#instructions">Instructions</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
# About The Project
This project was developed during the last semester of my 5 year course in Mechanical Engineering, in order for me to attain the Master's Degree. It was developed within the Augmanity Project, and in collaboration with Bosch Termotecnologia Aveiro.
The Augmanity project, developed at Bosch Termotecnologia Aveiro and the University of Aveiro, aims to digitize and optimize industrial processes in line with the concepts of Industry 4.0. This repository contains the implementation of an intelligent agent that utilizes machine learning algorithms, specifically autoencoders, to make predictions and detect anomalies upon energy consumption related variables. AutoML is used in the process of model building, and optimizes a series of the models' parameters, making sure that each model is suited for the behaviourial patterns of the specific variable. Beyond this processing capabilities, this project also implements an architecture that allows for data visualization (throguh a Grafana dashboard) and for the sending of automatic notifications regarding the processing, through a Telegram bot.


<!-- GETTING STARTED -->
# Implemented Architecture

The implemented architecture in the use case was as follows.
![Implemented Architecture](Implemented_Architecture.png)
The architecture is split in three main levels/layers:
  <li> <b> Physical layer </b> - comprised of the physical devices/equipments: the industrial compressor, connected to an energy analyzer (which gathers informations about 60 different energy related variables). Then a protocol  converter that allows the info to go to a gateway in the form of a Raspberry Pi;
  <li> <b> Digital layer </b> - layer where the data is gathered, organized, stored and processed. Data is sent via MQTT to Eclipse Hono. Hono, connected to Ditto via a Kafka connection, automatically updates the status of the digital twin (in Ditto) relative to the compressor. Everytim its status is updated, an SSE event is generated and the new data is sent to the SSE client, built with Python. The data is then filtered, and with an InfluxDB client also initiated, the new data is input to the Database. Then comes the processing. The assistant has a series of micro-services, eith three main ones: the dataset updator, the model builder and the intelligent assistant, which encompasses the anomaly detection and prediction functionalities;
  <li> <b> Users layer </b> - The layer where the user can interact with the architecture. In the use case, as it can be seen in the picture, the only interaction implemented was the grafana dashboard. However, in the code there is also the ability to send emails automatically and also the use of a Telegram bot, which automatically sends notifications to the user.
    
    
<!-- MAIN INSTRUCTIONS -->
# Instructions
    
<!-- Cloud2Edge setup + installation     -->
## Cloud2Edge installation + setup
    
<li> The steps to install Cloud2Edge in your machine can be found through <a href="https://www.eclipse.org/packages/packages/cloud2edge/installation/"><strong>here</strong></a>, in Cloud2Edge's official documentation; </li>
<li> Then, the steps that allow for the usage of Hono and Ditto, with the creation and management of digital twins, can be found in the <a href="https://github.com/zemaria2000/IntelligentAssistant/blob/main/c2e.txt">c2e.txt</a> text file. The contents of the text file were based on the official Ditto documentation, which can be accessed through <a href="https://www.eclipse.org/packages/packages/cloud2edge/tour/"><strong>here</strong></a>. </li>
    

<!-- Telegram bot generation  -->  

## How to create a Telegram bot

Even though only implemented at a simulated enviornment, in this work two Telegram bot interactions were developed: one where we, as the user, could send commands to the bot asking for information about the data (graphs, latest values, etc.) and he would then process the information and respond. And other where the bot would automatically send notifications regarding the anomaly detection task, which was intergated inside the "Assistant" service. Next I'll do a quick guide on how to create a Telegram bot for the first functionality, and then how to create a group for the second one.

<h3> Telegram bot generation </h3>

<ol>
  <li> Open the Telegram app (on your smartphone or computer) and search for <b>@BotFather</b>, choosing the first one that appears; </li>
  <li> Initialize the BotFather bot by typing <b>/start</b>; </li>
  <li> Create a new bot by typing <b>/newbot</b>; </li>
  <li> Then, just follow the steps that are sent to you by BotFather. You should create name your bot, create a username, and at the end a token is generated. <b>This token is essential and is the one used in the code to communicate with your bot</b>; </li>
</ol>

![Telegram bot generation](Bot.png)


<h3> Telegram group generation </h3>
  
<ol>
  <li> First access the Telegram menu and select the <b>New Group</b> option;</li>
  <li> Then add the members you wish to be part of the group, obviously including the bot previously generated;</li>
  <li> Name your group as you wish;</li>
  <li> Then, access the following url "https://api.telegram.org/bot'bot_token'/getUpdates" in your browser, <b>replacing the 'bot_token' for the one previously generated</b>. This will show you a JSON document, where you'll need to access the <b>chat object, and copy the group ID to the script where you want to use the bot. </li>
</ol>

<!-- Email connection -->  

## Gmail password generation


## Launching the services

You'll just need to access the <a href = https://github.com/zemaria2000/IntelligentAssistant/tree/main/Intelligent%20Assistant><strong>Intelligent Assistant</strong></a> directory and just launch the docker-compose. Each service has a requirements.txt file that is installed when the respective Dockerfile is ran. 






    
2. Mention all the steps needed to use the docker-compose and so on
3. Probably talk also about Telegram + gmail password generation

 
    


## Simulation:


<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the GPL License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

José Cação - josemaria@ua.pt

Project Link: [Intelligent Assistant](https://github.com/zemaria2000/IntelligentAssistant)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
- Professor José Paulo Santos - jps@ua.pt (Departamento de Engenharia Mecânica da Universidade de Aveiro)
- Professor Mário Antunes - mario.antunes@ua.pt (Instituto de Telecomunicações da Universidade de Aveiro)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
https://www.markdownguide.org/basic-syntax/#reference-style-links
<!-- [contributors-shield]: https://img.shields.io/github/contributors/TatianaResend/SPIF-A_v2.svg?style=for-the-badge -->
[contributors-url]: https://github.com/zemaria2000/IntelligentAssistant/contributors
<!-- [forks-shield]: https://img.shields.io/github/forks/TatianaResend/SPIF-A_v2.svg?style=for-the-badge -->
[forks-url]: https://github.com/zemaria2000/IntelligentAssistant/network/members
<!-- [stars-shield]: https://img.shields.io/github/stars/TatianaResend/SPIF-A_v2.svg?style=for-the-badge -->
[stars-url]: https://github.com/zemaria2000/IntelligentAssistant/stargazers
<!-- [issues-shield]: https://img.shields.io/github/issues/TatianaResend/SPIF-A_v2.svg?style=for-the-badge -->
[issues-url]: https://github.com/zemaria2000/IntelligentAssistant/issues
<!-- [license-shield]: https://img.shields.io/github/license/TatianaResend/SPIF-A_v2.svg?style=for-the-badge -->
[license-url]: https://github.com/zemaria2000/IntelligentAssistant/blob/master/LICENSE.txt
