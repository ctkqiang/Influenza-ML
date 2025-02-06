#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
社交媒体疫情数据采集系统
作者: 上饶满星科技有限公司 数据采集组
功能: 自动化采集社交媒体上的疫情相关信息
"""

import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime
import logging
import os
import json
from typing import List, Dict, Any

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="数据采集日志.log",
)


class SocialMediaScraper:
    """社交媒体数据采集器"""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def scrape_tweets(
        self, keywords: List[str], num_tweets: int, lang: str = "zh"
    ) -> List[Dict[str, Any]]:
        tweets_list: List[Dict[str, Any]] = []
        query: str = f"{' OR '.join(keywords)} lang:{lang}"

        logging.info(f"开始采集推文 - 关键词: {keywords}")

        try:
            for i, tweet in enumerate(
                sntwitter.TwitterSearchScraper(query).get_items()
            ):
                if i >= num_tweets:
                    break

                tweet_data: Dict[str, Any] = {
                    "采集时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "发布时间": tweet.date,
                    "推文ID": tweet.id,
                    "内容": tweet.content,
                    "用户名": tweet.user.username,
                    "转发数": tweet.retweetCount,
                    "点赞数": tweet.likeCount,
                    "回复数": tweet.replyCount,
                    "来源": tweet.source,
                    "地理位置": tweet.place.fullName if tweet.place else None,
                }
                tweets_list.append(tweet_data)

                if (i + 1) % 100 == 0:
                    logging.info(f"已采集 {i + 1} 条推文")

        except Exception as e:
            logging.error(f"数据采集过程中出现错误: {str(e)}")

        return tweets_list

        def save_data(self, data: List[Dict[str, Any]], keywords: List[str]) -> None:
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            keywords_str: str = "_".join(keywords)

            excel_path: str = os.path.join(
                self.output_dir, f"疫情数据_{keywords_str}_{timestamp}.xlsx"
            )
            df: pd.DataFrame = pd.DataFrame(data)
            df.to_excel(excel_path, index=False)

            json_path: str = os.path.join(
                self.output_dir, f"疫情数据_{keywords_str}_{timestamp}.json"
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logging.info(f"数据已保存至: {excel_path}")
            logging.info(f"数据备份至: {json_path}")


def main() -> None:
    keywords: List[str] = ["禽流感", "H5N1", "疫情", "outbreak"]
    num_tweets: int = 1000

    scraper: SocialMediaScraper = SocialMediaScraper()

    logging.info("开始数据采集任务")
    tweets: List[Dict[str, Any]] = scraper.scrape_tweets(keywords, num_tweets)

    scraper.save_data(tweets, keywords)
    logging.info("数据采集任务完成")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("程序被用户中断")
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
