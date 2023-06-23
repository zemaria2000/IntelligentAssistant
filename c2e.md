# 1. Initialization

Open a new terminal window, and put the following commands:
    
    RELEASE=c2e
    NS=cloud2edge
    ./setCloud2EdgeEnv.sh $RELEASE $NS
    eval "$(./setCloud2EdgeEnv.sh $RELEASE $NS)"

Note that you should first download 'setCloud2EdgeEnv.sh' from 'https://www.eclipse.org/packages/packages/cloud2edge/tour/'


# 2. Create a tenant (Hono)
The name of the tenant you wish to create should replace the <tenant> element
    
    curl -i -X POST http://${REGISTRY_IP}:${REGISTRY_PORT_HTTP}/v1/tenants/<tenant>


# 3. Create device (Hono)
It is recommended that the namespace and device are the same for Hono and then Ditto

    curl -i -X POST http://${REGISTRY_IP}:${REGISTRY_PORT_HTTP}/v1/devices/<tenant>/<namespace>:<device>


# 4. Create authentication (Hono)
The authentication keys (ID + password) are really important for the MQTT connection between gateway and Hono. To send data to Hono via gateway, we need
to make sure that the MQTT username is "<auth-id>@<tenant>" and the passowrd is "<pwd-plain>"

    curl -i -X PUT -H "Content-Type: application/json" --data '[
    {
    "type": "hashed-password",
    "auth-id": "<id>",
    "secrets": [{
        "pwd-plain": "<pass>"
    }]
    }]' http://${REGISTRY_IP}:${REGISTRY_PORT_HTTP}/v1/credentials/<tenant>/<namespace>:<device>



# 5. Define Hono tenant (just an env variable in the terminal)

    HONO_TENANT=<tenant>


# 6. Connect to Ditto API
    
    DITTO_DEVOPS_PWD=$(kubectl --namespace ${NS} get secret ${RELEASE}-ditto-gateway-secret -o jsonpath="{.data.devops-password}" | base64 --decode)


# 7. Establish connection between Hono and Ditto
There are numerous possibilities for the connection, such as AMQP or kafka. The following code creates a kafka connection between Hono and Ditto

    curl -i -X POST -u devops:${DITTO_DEVOPS_PWD} -H 'Content-Type: application/json' --data '{
        "targetActorSelection": "/system/sharding/connection",
        "headers": {
            "aggregate": false
        },
        "piggybackCommand": {
            "type": "connectivity.commands:createConnection",
            "connection": {
                "id": "eclipse-hono-via-kafka-augmanity",
                "name": "eclipse-hono-via-kafka",
                "connectionType": "kafka",
                "connectionStatus": "open",
                "uri": "ssl://ditto-c2e:verysecret@c2e-kafka-headless:9092",
                "sources": [
                    {
                        "addresses": [
                            "hono.telemetry.augmanity"
                        ],
                        "consumerCount": 1,
                        "authorizationContext": [
                            "pre-authenticated:hono-connection"
                        ],
                        "qos": 0,
                        "enforcement": {
                            "input": "{{ header:device_id }}",
                            "filters": [
                                "{{ entity:id }}"
                            ]
                        },
                        "headerMapping": {},
                        "payloadMapping": [],
                        "replyTarget": {
                            "enabled": true,
                            "address": "hono.command.augmanity/{{ thing:id }}",
                            "headerMapping": {
                                "device_id": "{{ thing:id }}",
                                "subject": "{{ header:subject | fn:default(topic:action-subject) | fn:default(topic:criterion) }}-response",
                                "correlation-id": "{{ header:correlation-id }}"
                            },
                            "expectedResponseTypes": [
                                "response",
                                "error"
                            ]
                        },
                        "acknowledgementRequests": {
                            "includes": [],
                            "filter": "fn:delete()"
                        },
                        "declaredAcks": []
                    },
                    {
                        "addresses": [
                            "hono.event.augmanity"
                        ],
                        "consumerCount": 1,
                        "authorizationContext": [
                            "pre-authenticated:hono-connection"
                        ],
                        "qos": 1,
                        "enforcement": {
                            "input": "{{ header:device_id }}",
                            "filters": [
                                "{{ entity:id }}"
                            ]
                        },
                        "headerMapping": {},
                        "payloadMapping": [],
                        "replyTarget": {
                            "enabled": true,
                            "address": "hono.command.augmanity/{{ thing:id }}",
                            "headerMapping": {
                                "device_id": "{{ thing:id }}",
                                "subject": "{{ header:subject | fn:default(topic:action-subject) | fn:default(topic:criterion) }}-response",
                                "correlation-id": "{{ header:correlation-id }}",
                                "expectedResponseTypes": [
                                    "response",
                                    "error"
                                ]
                            },
                            "acknowledgementRequests": {
                                "includes": []
                            },
                            "declaredAcks": []
                        }
                    },
                    {
                        "addresses": [
                            "hono.command_response.augmanity"
                        ],
                        "consumerCount": 1,
                        "authorizationContext": [
                            "pre-authenticated:hono-connection"
                        ],
                        "qos": 0,
                        "enforcement": {
                            "input": "{{ header:device_id }}",
                            "filters": [
                                "{{ entity:id }}"
                            ]
                        },
                        "headerMapping": {
                            "correlation-id": "{{ header:correlation-id }}",
                            "status": "{{ header:status }}"
                        },
                        "payloadMapping": [],
                        "replyTarget": {
                            "enabled": false,
                            "expectedResponseTypes": [
                                "response",
                                "error"
                            ]
                        },
                        "acknowledgementRequests": {
                            "includes": [],
                            "filter": "fn:delete()"
                        },
                        "declaredAcks": []
                    }
                ],
                "targets": [
                    {
                        "address": "hono.command.augmanity/{{ thing:id }}",
                        "authorizationContext": [
                            "pre-authenticated:hono-connection"
                        ],
                        "headerMapping": {
                            "device_id": "{{ thing:id }}",
                            "subject": "{{ header:subject | fn:default(topic:action-subject) }}",
                            "correlation-id": "{{ header:correlation-id }}",
                            "response-required": "{{ header:response-required }}"
                        },
                        "topics": [
                            "_/_/things/live/commands",
                            "_/_/things/live/messages"
                        ]
                    },
                    {
                        "address": "hono.command.augmanity/{{thing:id}}",
                        "authorizationContext": [
                            "pre-authenticated:hono-connection"
                        ],
                        "topics": [
                            "_/_/things/twin/events",
                            "_/_/things/live/events"
                        ],
                        "headerMapping": {
                            "device_id": "{{ thing:id }}",
                            "subject": "{{ header:subject | fn:default(topic:action-subject) }}",
                            "correlation-id": "{{ header:correlation-id }}"
                        }
                    }
                ],
                "specificConfig": {
                    "saslMechanism": "plain",
                    "bootstrapServers": "c2e-kafka:9092",
                    "groupId": "{{ connection:id }}"
                },
                "clientCount": 1,
                "failoverEnabled": true,
                "validateCertificates": true,
                "ca": "-----BEGIN CERTIFICATE-----\nMIICXzCCAgSgAwIBAgIUa42/FS599Wc7DdPDlQ2lKxqSXpEwCgYIKoZIzj0EAwIw\nUDELMAkGA1UEBhMCQ0ExDzANBgNVBAcMBk90dGF3YTEUMBIGA1UECgwLRWNsaXBz\nZSBJb1QxDTALBgNVBAsMBEhvbm8xCzAJBgNVBAMMAmNhMB4XDTIyMDYyMjA2MzM1\nMloXDTIzMDYyMjA2MzM1MlowUzELMAkGA1UEBhMCQ0ExDzANBgNVBAcMBk90dGF3\nYTEUMBIGA1UECgwLRWNsaXBzZSBJb1QxDTALBgNVBAsMBEhvbm8xDjAMBgNVBAMM\nBWthZmthMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEEr76VzDj41w4Q/v8xuBx\nMJJpkHZaVzFHUHv05G2Em7IGVXJX3YDxiPozV984gOOgjHjqlYpglA6WtWy6FPkG\n6aOBuDCBtTAdBgNVHQ4EFgQUipGlLxpw3qBqOGbmPvNSHl2BC8YwCwYDVR0PBAQD\nAgOoMB0GA1UdJQQWMBQGCCsGAQUFBwMBBggrBgEFBQcDAjBHBgNVHREEQDA+ghUq\nLmhvbm8ta2Fma2EtaGVhZGxlc3OCGiouaG9uby1rYWZrYS1oZWFkbGVzcy5ob25v\ngglsb2NhbGhvc3QwHwYDVR0jBBgwFoAUr/zExcFn/Jf7gFFB5oiwcFUb0QMwCgYI\nKoZIzj0EAwIDSQAwRgIhANxDsUydey3KmprMe2n2cmiMWXwJqag/h+KMoLrZk9S7\nAiEAspZFzsmxMQF8au/EYhNYj0WNC+8ppfclq+/305IdjYU=\n-----END CERTIFICATE-----"
            }
        }
    }' http://${DITTO_API_IP}:${DITTO_API_PORT_HTTP}/devops/piggyback/connectivity


# 8. Create a policy
If you wish to rename the policy, just change the name "default-policy" in the end

    curl -i -X PUT -u ditto:ditto -H 'Content-Type: application/json' --data '{
    "entries": {
        "DEFAULT": {
        "subjects": {
            "{{ request:subjectId }}": {
            "type": "Ditto user authenticated via nginx"
            }
        },
        "resources": {
            "thing:/": {
            "grant": ["READ", "WRITE"],
            "revoke": []
            },
            "policy:/": {
            "grant": ["READ", "WRITE"],
            "revoke": []
            },
            "message:/": {
            "grant": ["READ", "WRITE"],
            "revoke": []
            }
        }
        },
        "HONO": {
        "subjects": {
            "pre-authenticated:hono-connection": {
            "type": "Connection to Eclipse Hono"
            }
        },
        "resources": {
            "thing:/": {
            "grant": ["READ", "WRITE"],
            "revoke": []
            },
            "message:/": {
            "grant": ["READ", "WRITE"],
            "revoke": []
            }
        }
        }
    }
    }' http://${DITTO_API_IP}:${DITTO_API_PORT_HTTP}/api/2/policies/<namespace>:default-policy


# 9. Create a device (Ditto)
Once again, it is recommended that the <namespace> and <device> are the same as the ones created in Hono

    curl -i -X PUT -u ditto:ditto -H 'Content-Type: application/json' --data '{
    "policyId": "<namespace>:default-policy",
    "attributes": {
        "location": "Germany"
    },
    "features": {
    }
    }' http://${DITTO_API_IP}:${DITTO_API_PORT_HTTP}/api/2/things/<namespace>:<device>


# 10. Check oncoming data for the twin
After creating the digital twin, this curl allows us to check when new data arrives at Ditto

    curl --http2 -u ditto:ditto -H 'Accept:text/event-stream' -N http://$DITTO_API_IP:$DITTO_API_PORT_HTTP/api/2/things


# Final note
If we want to add a new device to the same tenant and using this particular Hono-Ditto connection, steps 2, 6 and 7 can be skipped