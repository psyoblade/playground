FROM python:3.6
MAINTAINER park.suhyuk@gmail.com
ADD ./examples/docker-agent /agent
ADD . /pommerman
RUN cd /pommerman && pip install .
EXPOSE 10080
VOLUME ["/Users/psyoblade/git/playground/models"]
ENV NAME Agent
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]
