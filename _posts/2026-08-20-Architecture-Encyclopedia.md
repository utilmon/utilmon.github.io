---
layout: post
title: Architecture Encyclopedia
date: 2026-08-20
description: Public documentation for the ArchitectureEncyclopedia Creator Upload Automation API client.
permalink: /blog/2026/Architecture-Encyclopedia/
tags: arts
categories: sample-posts
---

# ArchitectureEncyclopedia Creator Upload Automation

**ArchitectureEncyclopedia Creator Upload Automation** is a privately operated Python application used by the owner of Architecture Encyclopedia to upload original educational architecture videos to the owner's YouTube channel.

This API client is **not offered to the public** and does not provide third-party users with access to upload or manage videos.

[Privacy Policy](https://utilmon.github.io/blog/2026/Architecture-Encyclopedia-Privacy-Policy/) · [Terms of Service](https://utilmon.github.io/blog/2026/Architecture-Encyclopedia-Terms-of-Service/)

---

## YouTube API Integration

The application uses the **YouTube Data API v3** and Google OAuth 2.0 to authorize the owner's Google account and upload videos to the owner's channel.

<a href="https://www.youtube.com/" target="_blank" rel="noopener noreferrer">
  <img
    src="https://developers.google.com/static/youtube/images/developed-with-youtube-sentence-case-dark.png"
    alt="Developed with YouTube"
    width="240"
  >
</a>

The operator provides the video file and upload metadata, including:

- Video title
- Video description
- Privacy status (`public`, `unlisted`, or `private`)
- Audience designation, including whether the video is made for kids when applicable

The client uses the upload functionality of the YouTube Data API to publish the selected video and metadata to the authorized channel.

---

## API Client Details

| Item | Description |
| --- | --- |
| API Client | ArchitectureEncyclopedia Creator Upload Automation |
| Operator | Private individual / owner of Architecture Encyclopedia |
| Access | Private, internal use only |
| Platform | Local Python application |
| API | YouTube Data API v3 |
| Authorization | Google OAuth 2.0 |
| Primary purpose | Upload original Architecture Encyclopedia videos |
| Intended viewers of uploaded videos | General public |
| Third-party user accounts | None |

---

## How Authorization Works

1. The operator runs the Python upload application.
2. Google OAuth 2.0 is used to authorize access to the operator's own Google/YouTube account.
3. The application receives authorization required to upload videos.
4. The operator selects the video and upload metadata.
5. The application uploads the video to the authorized YouTube channel using the YouTube Data API.
6. Access can be revoked through the operator's Google Account permissions.

The application is intended to request only the permissions necessary for its upload functionality.

---

## Privacy and Terms

Use of this API client is governed by the following documents:

- [Privacy Policy](https://utilmon.github.io/blog/2026/Architecture-Encyclopedia-Privacy-Policy/)
- [Terms of Service](https://utilmon.github.io/blog/2026/Architecture-Encyclopedia-Terms-of-Service/)

The Privacy Policy describes the client's use of YouTube API Services, Google authorization, data handling, and revocation/deletion options.

---

## About Architecture Encyclopedia

Architecture Encyclopedia creates original educational videos explaining architecture, buildings, structural systems, architectural history, and related topics.

The upload automation exists solely to streamline publishing of content created and controlled by the channel owner.

---

*ArchitectureEncyclopedia Creator Upload Automation is an independent application and is not endorsed by or affiliated with YouTube or Google.*
