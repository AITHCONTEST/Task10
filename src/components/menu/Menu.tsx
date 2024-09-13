import Favs from "./Favs";
import styles from "./../styles/menu.module.scss"
import React from "react";


interface MenuProps {
  tap: () => void;
  style: React.CSSProperties;
}

const Menu = ({tap, style}:MenuProps) => {
   

    return(
        <div className={styles.menu}  style={style}>
            <a href="https://cloud.uriit.ru/s/MWo10yvg1a3OmtE" target="_blank">Скачать клавиатуру!</a>
            <Favs tap={tap}/>
        </div>
    );
};

export default Menu;